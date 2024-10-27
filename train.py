import argparse
import math
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch import distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import models
from pytorch_msssim import ms_ssim

from models import LICAutoencoder
from torch.utils.tensorboard import SummaryWriter   
from PIL import Image
import os

from time import time

torch.set_num_threads(10)

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.)

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, beta = 1, alpha = 1, lmbda=1e-2, type='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.beta = beta
        self.alpha = alpha
        self.type = type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["cca_loss"] = (
            torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2)) - 
            torch.log(output["aux_likelihoods"]["y_cca"]).sum() / (-math.log(2))
        ) / num_pixels
        out["aux2_loss"] = torch.sum(output["aux_likelihoods"]["y_cca"] * torch.log(output["aux_likelihoods"]["y_aux"])) / (-math.log(2) * num_pixels)
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.type == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + self.beta * out["bpp_loss"] + self.alpha * out["cca_loss"] + out["aux2_loss"]
        else:
            out['ms_ssim_loss'] = compute_msssim(output["x_hat"], target)
            out["loss"] = self.lmbda * (1 - out['ms_ssim_loss']) + self.beta * out["bpp_loss"] + self.alpha * out["cca_loss"] + out["aux2_loss"]

        return out

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, log_path, train_sampler, type='mse'
):
    model.train()
    device = next(model.parameters()).device

    if torch.cuda.device_count() > 1:
        train_sampler.set_epoch(epoch)

    lastime1 = time()
    lastime100 = lastime1
    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.module.aux_loss() if torch.cuda.device_count() > 1 else model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()
        
        if i % 100 == 0:
            train_logging = open(log_path+"/log.txt", "a")
            print("100 times timecost: ", time() - lastime100)
            lastime100 = time()
            if type == 'mse':
                result = (
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f'\tCCA loss: {out_criterion["cca_loss"].item():.3f} |'
                    f"\tAux loss: {aux_loss.item():.2f} |"
                    f'\tAux2 loss: {out_criterion["aux2_loss"].item():.3f}\n'
                )
                print(result)
                train_logging.write(result)
            else:
                result = (
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMS_SSIM loss: {out_criterion["ms_ssim_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f'\tCCA loss: {out_criterion["cca_loss"].item():.3f} |'
                    f"\tAux loss: {aux_loss.item():.2f} |"
                    f'\tAux2 loss: {out_criterion["aux2_loss"].item():.3f}\n'
                )
                print(result)
                train_logging.write(result)
            train_logging.close()


def test_epoch(epoch, test_dataloader, model, criterion, log_path, type='mse'):
    model.eval()
    device = next(model.parameters()).device
    train_logging = open(log_path+"/log.txt", "a")
    if type == 'mse':
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        mse_loss = AverageMeter()
        cca_loss = AverageMeter()
        aux_loss = AverageMeter()
        aux2_loss = AverageMeter()

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d)
                out_criterion = criterion(out_net, d)

                aux_loss.update(model.module.aux_loss() if torch.cuda.device_count() > 1 else model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                mse_loss.update(out_criterion["mse_loss"])
                cca_loss.update(out_criterion["cca_loss"])
                aux2_loss.update(out_criterion["aux2_loss"])

        result = (
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMSE loss: {mse_loss.avg:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
            f"\tCCA loss: {cca_loss.avg:.3f} |"
            f"\tAux loss: {aux_loss.avg:.2f} |"
            f"\tAux2 loss: {aux2_loss.avg:.3f}\n"
        )
        print(result)
        train_logging.write(result)

    else:
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        ms_ssim_loss = AverageMeter()
        cca_loss = AverageMeter()
        aux_loss = AverageMeter()
        aux2_loss = AverageMeter()

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d)
                out_criterion = criterion(out_net, d)

                aux_loss.update(model.module.aux_loss() if torch.cuda.device_count() > 1 else model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"])
                cca_loss.update(out_criterion["cca_loss"])
                aux2_loss.update(out_criterion["aux2_loss"])

        result = (
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMS_SSIM loss: {ms_ssim_loss.avg:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
            f"\tCCA loss: {cca_loss.avg:.3f} |"
            f"\tAux loss: {aux_loss.avg:.2f} |"
            f"\tAux2 loss: {aux2_loss.avg:.3f}\n"
        )
        print(result)
        train_logging.write(result)
    train_logging.close()
    return loss.avg, (mse_loss.avg if type == 'mse' else ms_ssim_loss.avg) + bpp_loss.avg

def test_compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def test_compute_msssim(a, b):
    return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

def test_compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(
            [torch.log(out_net["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels)] +
            [torch.log(likelihood).sum() / (-math.log(2) * num_pixels)
            for likelihood in out_net["likelihoods"]["y"]]
        ).item()

def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

def save_checkpoint(state, is_best, epoch, save_path, filename):
    torch.save(state, save_path + "ckpttars/" + "checkpoint_latest.pth.tar")
    if epoch % 5 == 0:
        torch.save(state, filename)
    if is_best:
        torch.save(state, save_path + "ckpttars/" + "checkpoint_best.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("-m", "--model", default="bmshj2018-factorized", choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("-e", "--epochs", default=50, type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr", "--learning-rate", default=1e-4, type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n", "--num-workers", type=int, default=10,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda", dest="lmbda", type=float, default=0.05,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--beta", dest="beta", type=float, default=1,
        help="Bpp parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--alpha", dest="alpha", type=float, default=1,
        help="CCA-loss parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=8,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate", default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size", type=int, nargs=2, default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=100, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm", default=1.0, type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--type", type=str, default='mse', help="loss type", choices=['mse', "ms-ssim"])
    parser.add_argument("--save_path", type=str, help="save_path")
    parser.add_argument("--lr_epoch", nargs='+', type=int)
    parser.add_argument("--continue_train", action="store_true", default=True)
    parser.add_argument(
        "-ch", "--channel", type=int, default=320,
        help="M, the channel of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--ae_dim", nargs='+', type=int, default=[192, 224, 256],
        help="dimensions of VAE backbone (default: %(default)s)",
    )
    parser.add_argument(
        "--em_dim", type=int, default=224,
        help="dimension of entropy model (default: %(default)s)",
    )
    parser.add_argument(
        "--ae_layers", nargs='+', type=int, default=[4, 4, 4],
        help="the numbers of layers of VAE backbone (default: %(default)s)",
    )
    parser.add_argument(
        "--em_layers", type=int, default=4,
        help="the number of layers of entropy model (default: %(default)s)",
    )
    parser.add_argument(
        "-prop", "--prop-slices", type=float, default=1.7,
        help="the channel proportion of grouped slices of latent variable (default: %(default)s)",
    )
    parser.add_argument("--cca_training", default=True, action="store_true", help="Enable CCA training")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    type = args.type
    save_path = os.path.join(args.save_path, str(args.beta)) + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path + "tensorboard/")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    writer = SummaryWriter(save_path + "tensorboard/")

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )
    
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    else: 
        device = "cuda"
    print(device)

    k = args.prop_slices
    net = LICAutoencoder(M = args.channel, 
                         prop_slices = [1, 2**k, 3**k, 4**k, 5**k], 
                         ae_dim = args.ae_dim, 
                         em_dim = args.em_dim, 
                         ae_layers = args.ae_layers,
                         em_layers = args.em_layers,
                         cca_training = args.cca_training)
    net = net.to(device)

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    if args.cuda and torch.cuda.device_count() > 1:
        net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)
    
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler if torch.cuda.device_count() > 1 else None,
        shuffle=True if torch.cuda.device_count() == 1 else None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        sampler=test_sampler if torch.cuda.device_count() > 1 else None,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    optimizer, aux_optimizer = configure_optimizers(net, args)
    milestones = args.lr_epoch
    print("milestones: ", milestones)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        net.load_state_dict(checkpoint["state_dict"])
        if args.continue_train:
            last_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=last_epoch-1)

    for group in optimizer.param_groups: group['lr'] = args.learning_rate
    
    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        if epoch < args.lr_epoch[0]:
            tmpbeta = 0.3 if args.beta < 7.0 else 2
            criterion = RateDistortionLoss(lmbda=args.lmbda, beta=tmpbeta, alpha=args.alpha, type=type)
            print(f"initial beta: {tmpbeta} (objective beta: {args.beta})")
        else:
            criterion = RateDistortionLoss(lmbda=args.lmbda, beta=args.beta, alpha=args.alpha, type=type) 
            print(f"beta: {args.beta}")

        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            save_path,
            train_sampler if torch.cuda.device_count() > 1 else None,
            type
        )
        loss, rec_loss = test_epoch(epoch, test_dataloader, net, criterion, save_path, type)

        global_rank = dist.get_rank() if torch.cuda.device_count() > 1 else 0
        if global_rank == 0:
            writer.add_scalar('test_loss', loss, epoch)
            writer.add_scalar('rec_loss', rec_loss, epoch)

        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save and global_rank == 0:
            os.makedirs(save_path + "ckpttars/", exist_ok=True)
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                epoch,
                save_path,
                save_path + "ckpttars/" + str(epoch) + "_checkpoint.pth.tar",
            )

if __name__ == "__main__":
    main(sys.argv[1:])