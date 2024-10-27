from compressai.entropy_models import EntropyBottleneck, GaussianConditional
import torch
import torch.nn as nn
from torch import Tensor
import math

def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )

def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

class NAFBlock(nn.Module):
    def __init__(self, dim, inter_dim=None):
        super().__init__()

        self.dim = inter_dim if inter_dim is not None else dim

        dw_channel = self.dim<<1
        ffn_channel = self.dim<<1

        self.dwconv = nn.Sequential(
            nn.Conv2d(self.dim, dw_channel, 1),
            nn.Conv2d(dw_channel, dw_channel, 3, 1, padding=1, groups=dw_channel)
        )
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1)
        )
        self.FFN = nn.Sequential(
            nn.Conv2d(self.dim, ffn_channel, 1),
            SimpleGate(),
            nn.Conv2d(ffn_channel>>1, self.dim, 1)
        )
        
        self.norm1 = LayerNorm2d(self.dim)
        self.norm2 = LayerNorm2d(self.dim)
        self.conv1 = nn.Conv2d(dw_channel>>1, self.dim, 1)

        self.beta = nn.Parameter(torch.zeros((1,self.dim,1,1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1,self.dim,1,1)), requires_grad=True)
        
        self.in_conv = conv(dim, inter_dim, kernel_size=1, stride=1) if inter_dim is not None else nn.Identity()
        self.out_conv = conv(inter_dim, dim, kernel_size=1, stride=1) if inter_dim is not None else nn.Identity()

    def forward(self, x):
        x = self.in_conv(x)
        identity = x
        x = self.norm1(x)

        x = self.dwconv(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv1(x)

        out = identity + x * self.beta
        identity = out

        out = self.norm2(out)
        out = self.FFN(out)

        out = identity + out * self.gamma

        out = self.out_conv(out)
        return out

class blocks(nn.Module):
    def __init__(self, input_dim, output_dim, layers=4, inter_dim=128) -> None:
        super().__init__()
        
        self.layers = layers
        self.blocks = nn.ModuleList(NAFBlock(inter_dim) for _ in range(self.layers))
        
        self.in_conv = conv(input_dim, inter_dim, kernel_size=1, stride=1)
        self.out_conv = conv(inter_dim, output_dim, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.in_conv(x)
        identity = out
        for i in range(self.layers): 
            out = self.blocks[i](out)
        out += identity
        out = self.out_conv(out)
        return out
    
class AuxEntropyModel(nn.Module):
    def __init__(
        self, 
        M = 320, 
        prop_slices = [1, 1, 1, 1, 1], 
        em_dim = 224, 
        em_layers = 4,
    ):
        super().__init__()

        if len(prop_slices) == 1:
            self.num_slices = prop_slices[0]
            prop_slices = [1 for _ in range(self.num_slices)]
        else: self.num_slices = len(prop_slices)
        self.size_slices = list(math.floor(1.0 * M / sum(prop_slices) * prop) for prop in prop_slices) 
        self.size_slices[self.num_slices - 1] += M - sum(self.size_slices)

        self.mean_NAF_transforms = nn.ModuleList(
            blocks(M + sum(self.size_slices[:(i-1 if i-1 > 0 else 0)]), M + sum(self.size_slices[:(i-1 if i-1 > 0 else 0)]), layers=em_layers, inter_dim=em_dim) for i in range(self.num_slices)
        )
        
        self.scale_NAF_transforms = nn.ModuleList(
            blocks(M + sum(self.size_slices[:(i-1 if i-1 > 0 else 0)]), M + sum(self.size_slices[:(i-1 if i-1 > 0 else 0)]), layers=em_layers, inter_dim=em_dim) for i in range(self.num_slices)
        )

        self.mean_cc_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + sum(self.size_slices[:(i-1 if i-1 > 0 else 0)]), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, self.size_slices[i], stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.scale_cc_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + sum(self.size_slices[:(i-1 if i-1 > 0 else 0)]), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, self.size_slices[i], stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        if self.num_slices > 2:
            self.lrp_transforms = nn.ModuleList(
                nn.Sequential(
                    conv(M + self.size_slices[i] + sum(self.size_slices[:(i-1 if i-1 > 0 else 0)]), 224, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(224, 128, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(128, self.size_slices[i], stride=1, kernel_size=3),
                ) for i in range(self.num_slices - 2)
            )

        self.y_entropy_bottleneck = EntropyBottleneck(M)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, y, latent_scales, latent_means):
        y_hat_slices = []
        y_likelihood = []

        _, y_aux_likelihoods = self.y_entropy_bottleneck(y)
        y_slices = y.split(self.size_slices, 1)
        
        for slice_index, y_slice in enumerate(y_slices):
            support_slices = y_hat_slices[:(slice_index-1)] if slice_index-1 > 0 else []
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.mean_NAF_transforms[slice_index](mean_support)
            mu = self.mean_cc_transforms[slice_index](mean_support)
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.scale_NAF_transforms[slice_index](scale_support)
            scale = self.scale_cc_transforms[slice_index](scale_support)
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            
            if self.num_slices > 2 and slice_index < self.num_slices - 2:
                y_hat_slice = ste_round(y_slice - mu) + mu
                lrp_support = torch.cat([mean_support, y_hat_slice], dim=1) 
                lrp = self.lrp_transforms[slice_index](lrp_support)
                lrp = 0.5 * torch.tanh(lrp)
                y_hat_slice += lrp
                y_hat_slices.append(y_hat_slice)

        y_likelihoods = torch.cat(y_likelihood, dim=1)

        return y_aux_likelihoods, y_likelihoods