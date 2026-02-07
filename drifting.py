#!/usr/bin/env python3
"""
Drifting Models for Image Generation
====================================

A practical image-generation script based on "Generative Modeling via Drifting"
(arXiv:2602.04770). This keeps the same core training recipe as the reference:

  loss = || f(eps) - stopgrad(f(eps) + V) ||^2

where V is computed by a kernelized drifting field using positive (data) and
negative (generated) samples.

Supported datasets:
- FashionMNIST (auto-download)
- ImageNet (either torchvision ImageNet layout or ImageFolder train/ layout)

Examples:
  python drifting.py --dataset fashionmnist --data-root ./data --epochs 5
  python drifting.py --dataset imagenet --data-root /path/to/imagenet --image-size 64
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import time
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torchvision.transforms import InterpolationMode
from torchvision.utils import make_grid, save_image

try:
    from torchvision.models import ResNet18_Weights
except Exception:  # pragma: no cover - old torchvision fallback
    ResNet18_Weights = None


LOGGER = logging.getLogger("drifting")


def setup_logging(console: Console) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    root.addHandler(RichHandler(console=console, rich_tracebacks=True, show_path=False))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def compute_drift(
    x: Tensor,
    y_pos: Tensor,
    y_neg: Tensor,
    temp: float = 0.05,
    eps: float = 1e-12,
) -> Tensor:
    """
    Compute V(x) with the doubly-normalized affinities from the paper appendix.
    """
    n, n_pos = x.shape[0], y_pos.shape[0]

    dist_pos = torch.cdist(x, y_pos)
    dist_neg = torch.cdist(x, y_neg)

    # Ignore self-interaction when negatives are the generated batch itself.
    if x.shape == y_neg.shape and x.data_ptr() == y_neg.data_ptr():
        dist_neg = dist_neg + torch.eye(n, device=x.device, dtype=x.dtype) * 1e6

    logit = torch.cat([-dist_pos / temp, -dist_neg / temp], dim=1)

    a_row = logit.softmax(dim=-1)  # normalization over y
    a_col = logit.softmax(dim=-2)  # extra normalization over x
    a = torch.sqrt((a_row * a_col).clamp_min(eps))

    a_pos = a[:, :n_pos]
    a_neg = a[:, n_pos:]

    w_pos = a_pos * a_neg.sum(dim=1, keepdim=True)
    w_neg = a_neg * a_pos.sum(dim=1, keepdim=True)

    return w_pos @ y_pos - w_neg @ y_neg


def normalize_feature_space(
    x_feat: Tensor,
    pos_feat: Tensor,
    neg_feat: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Feature normalization from appendix: average pairwise distance ~ sqrt(C).
    """
    c = x_feat.shape[-1]
    with torch.no_grad():
        merged = torch.cat([pos_feat, neg_feat], dim=0)
        s = torch.cdist(x_feat, merged).mean() / math.sqrt(float(c))
        s = s.clamp_min(eps)
    return x_feat / s, pos_feat / s, neg_feat / s


def compute_feature_drift(
    x_feat: Tensor,
    pos_feat: Tensor,
    temperatures: Sequence[float],
    *,
    norm_mode: str = "batch",
    ema_state: dict[str, float] | None = None,
    ema_prefix: str = "main",
    ema_decay: float = 0.99,
    eps: float = 1e-6,
) -> tuple[Tensor, float, float]:
    """
    Compute aggregated feature-space drift with optional multi-temperature setup.

    Returns:
      total_drift, raw_drift_sq, normed_drift_sq
    """
    x_norm, pos_norm, neg_norm = normalize_feature_space(x_feat, pos_feat, x_feat, eps)
    c = x_feat.shape[-1]
    temp_scale = math.sqrt(float(c))

    total = torch.zeros_like(x_norm)
    raw_vals: list[float] = []
    norm_vals: list[float] = []
    for tau in temperatures:
        tau_eff = max(float(tau), eps) * temp_scale
        v = compute_drift(x_norm, pos_norm, neg_norm, temp=tau_eff)

        with torch.no_grad():
            raw_sq = v.pow(2).sum(dim=-1).mean()
            raw_vals.append(raw_sq.item())
            base = raw_sq / float(c)
            if norm_mode == "batch":
                lam_sq = base
            elif norm_mode == "ema":
                if ema_state is None:
                    raise ValueError("ema_state must be provided when norm_mode='ema'.")
                key = f"{ema_prefix}|c={c}|tau={float(tau):g}"
                prev = ema_state.get(key)
                if prev is None:
                    ema_val = base.item()
                else:
                    ema_val = float(ema_decay) * prev + (1.0 - float(ema_decay)) * base.item()
                ema_state[key] = ema_val
                lam_sq = torch.tensor(ema_val, device=v.device, dtype=v.dtype)
            else:
                raise ValueError(f"Unsupported norm_mode: {norm_mode}")
            lam = torch.sqrt(lam_sq.clamp_min(eps))

        v_norm = v / lam
        with torch.no_grad():
            norm_vals.append(v_norm.pow(2).sum(dim=-1).mean().item())
        total = total + v_norm
    raw_drift_sq = sum(raw_vals) / max(1, len(raw_vals))
    normed_drift_sq = sum(norm_vals) / max(1, len(norm_vals))
    return total, raw_drift_sq, normed_drift_sq


def drifting_loss_from_features(
    fake_feat: Tensor,
    real_feat: Tensor,
    temperatures: Sequence[float],
    *,
    norm_mode: str,
    ema_state: dict[str, float],
    ema_prefix: str,
    ema_decay: float,
    eps: float = 1e-6,
) -> tuple[Tensor, float, float]:
    """
    stopgrad( fake_feat + V ) target, gradients only through fake_feat.
    """
    with torch.no_grad():
        drift, raw_drift_sq, normed_drift_sq = compute_feature_drift(
            fake_feat.detach(),
            real_feat.detach(),
            temperatures,
            norm_mode=norm_mode,
            ema_state=ema_state,
            ema_prefix=ema_prefix,
            ema_decay=ema_decay,
            eps=eps,
        )
        target = (fake_feat + drift).detach()

    loss_per_sample = (fake_feat - target).pow(2).mean(dim=-1)
    return loss_per_sample, raw_drift_sq, normed_drift_sq


def feature_moment_gap(fake_feat: Tensor, real_feat: Tensor) -> float:
    """
    Distribution mismatch proxy in feature space (independent of drift normalization).
    """
    with torch.no_grad():
        fake_mean = fake_feat.mean(dim=0)
        real_mean = real_feat.mean(dim=0)
        fake_std = fake_feat.std(dim=0, unbiased=False)
        real_std = real_feat.std(dim=0, unbiased=False)
        mean_mse = F.mse_loss(fake_mean, real_mean)
        std_mse = F.mse_loss(fake_std, real_std)
    return (mean_mse + std_mse).item()


def as_feature_list(feats: Tensor | Sequence[Tensor]) -> list[Tensor]:
    if torch.is_tensor(feats):
        return [feats]
    return list(feats)


class ConvGenerator(nn.Module):
    """
    Small upsampling generator: z -> image.
    """

    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        image_size: int,
        base_channels: int = 64,
        num_classes: int = 0,
    ):
        super().__init__()
        if image_size < 8 or not is_power_of_two(image_size):
            raise ValueError("image_size must be a power of two and >= 8.")

        num_upsamples = int(math.log2(image_size // 4))
        if 4 * (2**num_upsamples) != image_size:
            raise ValueError("image_size must be divisible by 4.")

        self.latent_dim = latent_dim
        self.num_classes = max(0, int(num_classes))
        self.class_embed = (
            nn.Embedding(self.num_classes, latent_dim) if self.num_classes > 0 else None
        )
        self.start_channels = base_channels * min(8, 2**num_upsamples)
        self.project = nn.Linear(latent_dim, self.start_channels * 4 * 4)

        layers: list[nn.Module] = []
        channels = self.start_channels
        current_size = 4
        while current_size < image_size:
            next_channels = max(base_channels, channels // 2)
            groups = 8 if next_channels % 8 == 0 else 1
            layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(channels, next_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(groups, next_channels),
                    nn.SiLU(inplace=True),
                ]
            )
            channels = next_channels
            current_size *= 2

        layers.extend([nn.Conv2d(channels, out_channels, kernel_size=3, padding=1), nn.Tanh()])
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, z: Tensor, labels: Tensor | None = None) -> Tensor:
        if self.class_embed is not None:
            if labels is None:
                raise ValueError("labels are required when class conditioning is enabled.")
            z = z + self.class_embed(labels)
        h = self.project(z).view(z.shape[0], self.start_channels, 4, 4)
        return self.net(h)

    @torch.no_grad()
    def generate(
        self, n: int, device: torch.device, labels: Tensor | None = None
    ) -> Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        return self(z, labels=labels)


class PyramidFeatureExtractor(nn.Module):
    """
    Lightweight fixed feature extractor:
    - multi-scale pooled intensities
    - local contrast map
    """

    def __init__(self, pool_sizes: Sequence[int] = (16, 8, 4), contrast_size: int = 8):
        super().__init__()
        self.pool_sizes = tuple(pool_sizes)
        self.contrast_size = contrast_size

    def forward(self, x: Tensor) -> Tensor:
        feats = []
        for size in self.pool_sizes:
            pooled = F.adaptive_avg_pool2d(x, output_size=(size, size))
            feats.append(pooled.flatten(start_dim=1))

        local_contrast = x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        contrast = F.adaptive_avg_pool2d(
            local_contrast, output_size=(self.contrast_size, self.contrast_size)
        )
        feats.append(contrast.flatten(start_dim=1))

        return torch.cat(feats, dim=1)


class PixelFeatureExtractor(nn.Module):
    """
    Pixel-space features to anchor fine detail and reduce feature-space shortcuts.
    """

    def __init__(self, size: int = 16):
        super().__init__()
        self.size = size

    def forward(self, x: Tensor) -> Tensor:
        if self.size > 0:
            x = F.interpolate(
                x, size=(self.size, self.size), mode="bilinear", align_corners=False
            )
        return x.flatten(start_dim=1)


class ResNet18FeatureExtractor(nn.Module):
    """
    Frozen ResNet18 feature encoder for drift loss.
    """

    def __init__(self, pretrained: bool = True, multi_scale: bool = True):
        super().__init__()
        self.multi_scale = multi_scale
        weights = None
        if pretrained and ResNet18_Weights is not None:
            weights = ResNet18_Weights.DEFAULT

        try:
            model = resnet18(weights=weights)
        except Exception as exc:
            if pretrained:
                LOGGER.warning(
                    "Could not load pretrained ResNet18 (%s). Falling back to random init.",
                    exc,
                )
            model = resnet18(weights=None)

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def train(self, mode: bool = True) -> "ResNet18FeatureExtractor":
        super().train(False)
        return self

    def forward(self, x: Tensor) -> Tensor | list[Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[-1] < 64:
            x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        feats = []
        for f in (f1, f2, f3, f4):
            pooled = F.adaptive_avg_pool2d(f, output_size=(1, 1)).flatten(start_dim=1)
            feats.append(F.normalize(pooled, dim=-1))

        if self.multi_scale:
            return feats
        return feats[-1]


def build_feature_extractor(
    name: str, dataset_name: str, use_pretrained: bool
) -> tuple[nn.Module, str]:
    resolved = name
    if name == "auto":
        resolved = "pyramid" if dataset_name == "fashionmnist" else "resnet18"

    if resolved == "pyramid":
        return PyramidFeatureExtractor(), resolved
    if resolved == "resnet18":
        return ResNet18FeatureExtractor(pretrained=use_pretrained, multi_scale=True), resolved
    raise ValueError(f"Unsupported feature extractor: {name}")


def image_transform(dataset_name: str, image_size: int) -> transforms.Compose:
    if dataset_name == "fashionmnist":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    if dataset_name == "imagenet":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.6, 1.0), interpolation=InterpolationMode.BICUBIC, antialias=True
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_dataset(
    dataset_name: str, data_root: Path, image_size: int
) -> tuple[torch.utils.data.Dataset, int, int]:
    transform = image_transform(dataset_name, image_size)

    if dataset_name == "fashionmnist":
        ds = datasets.FashionMNIST(
            root=str(data_root), train=True, transform=transform, download=True
        )
        return ds, 1, len(ds.classes)

    if dataset_name == "imagenet":
        train_dir = data_root / "train"
        if train_dir.is_dir():
            ds = datasets.ImageFolder(root=str(train_dir), transform=transform)
            return ds, 3, len(ds.classes)

        try:
            ds = datasets.ImageNet(root=str(data_root), split="train", transform=transform)
            classes = len(getattr(ds, "classes", [])) or 1000
            return ds, 3, classes
        except Exception as exc:
            raise RuntimeError(
                "ImageNet path not found. Use --data-root that either contains "
                "'train/<class_name>/*' (ImageFolder) or official torchvision "
                "ImageNet metadata layout."
            ) from exc

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )


def unpack_batch(batch: object) -> tuple[Tensor, Tensor | None]:
    if isinstance(batch, (tuple, list)):
        images = batch[0]
        labels = batch[1] if len(batch) > 1 and torch.is_tensor(batch[1]) else None
        return images, labels
    if torch.is_tensor(batch):
        return batch, None
    raise TypeError(f"Unsupported batch type: {type(batch)}")


@torch.no_grad()
def save_samples(
    generator: ConvGenerator,
    out_path: Path,
    n_samples: int,
    nrow: int,
    latent_dim: int,
    device: torch.device,
    fixed_z: Tensor | None = None,
    fixed_labels: Tensor | None = None,
) -> None:
    was_training = generator.training
    generator.eval()
    z = fixed_z if fixed_z is not None else torch.randn(n_samples, latent_dim, device=device)
    labels = None
    if fixed_labels is not None:
        labels = fixed_labels[: z.shape[0]].to(device)
    images = generator(z, labels=labels).cpu()
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    save_image(grid, str(out_path))
    if was_training:
        generator.train()


def save_checkpoint(
    generator: ConvGenerator,
    optimizer: torch.optim.Optimizer,
    out_path: Path,
    args: argparse.Namespace,
    step: int,
    epoch: int,
) -> None:
    state = {
        "generator": generator.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "args": vars(args),
    }
    torch.save(state, str(out_path))


def render_config_table(
    console: Console,
    args: argparse.Namespace,
    dataset_len: int,
    num_classes: int,
    feature_name: str,
    device: torch.device,
    run_dir: Path,
) -> None:
    table = Table(title="Drifting Image Training")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("dataset", args.dataset)
    table.add_row("data_root", str(args.data_root))
    table.add_row("dataset_size", f"{dataset_len:,}")
    table.add_row("image_size", str(args.image_size))
    table.add_row("batch_size", str(args.batch_size))
    table.add_row("num_classes", str(num_classes))
    table.add_row("class_conditioning", str(args.class_conditioning))
    table.add_row("epochs", str(args.epochs))
    table.add_row("steps_per_epoch", str(args.steps_per_epoch if args.steps_per_epoch > 0 else "full"))
    table.add_row("max_steps", str(args.max_steps if args.max_steps > 0 else "none"))
    table.add_row("latent_dim", str(args.latent_dim))
    table.add_row("sample_seed", str(args.sample_seed))
    table.add_row("feature_extractor", feature_name)
    table.add_row("drift_norm_mode", args.drift_norm_mode)
    table.add_row("drift_ema_decay", f"{args.drift_ema_decay:g}")
    table.add_row("pixel_drift_weight", f"{args.pixel_drift_weight:g}")
    table.add_row("pixel_drift_size", str(args.pixel_drift_size))
    table.add_row("temperatures", ",".join(f"{t:g}" for t in args.temperatures))
    table.add_row("device", str(device))
    table.add_row("run_dir", str(run_dir))
    console.print(table)


def train(args: argparse.Namespace) -> None:
    console = Console()
    setup_logging(console)
    seed_everything(args.seed)

    device = get_device(args.device)

    data_root = Path(args.data_root).expanduser()
    out_root = Path(args.output_dir).expanduser()
    run_dir = out_root / f"{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}"
    samples_dir = run_dir / "samples"
    checkpoints_dir = run_dir / "checkpoints"
    samples_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    dataset, out_channels, num_classes = build_dataset(args.dataset, data_root, args.image_size)
    loader = build_dataloader(dataset, args.batch_size, args.num_workers, device)
    cond_num_classes = num_classes if args.class_conditioning else 0

    feature_extractor, feature_name = build_feature_extractor(
        args.feature_backbone, args.dataset, args.pretrained_features
    )
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    for p in feature_extractor.parameters():
        p.requires_grad_(False)

    generator = ConvGenerator(
        latent_dim=args.latent_dim,
        out_channels=out_channels,
        image_size=args.image_size,
        base_channels=args.base_channels,
        num_classes=cond_num_classes,
    ).to(device)
    pixel_feature_extractor = PixelFeatureExtractor(size=args.pixel_drift_size).to(device)

    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    drift_ema_state: dict[str, float] = {}
    sample_noise_generator = torch.Generator(device="cpu")
    sample_noise_generator.manual_seed(args.sample_seed)
    fixed_sample_z = torch.randn(
        args.sample_count, args.latent_dim, generator=sample_noise_generator
    )
    if device.type != "cpu":
        fixed_sample_z = fixed_sample_z.to(device)
    fixed_sample_labels = None
    if cond_num_classes > 0:
        fixed_sample_labels = torch.arange(args.sample_count, dtype=torch.long) % cond_num_classes

    global_step = 0
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        generator.load_state_dict(ckpt["generator"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = int(ckpt.get("step", 0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        LOGGER.info("Resumed from %s at step=%d epoch=%d", args.resume, global_step, start_epoch)

    render_config_table(
        console, args, len(dataset), num_classes, feature_name, device, run_dir
    )

    steps_per_epoch = (
        len(loader) if args.steps_per_epoch <= 0 else min(args.steps_per_epoch, len(loader))
    )
    max_steps = args.max_steps if args.max_steps > 0 else None

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TextColumn("loss={task.fields[loss]:.5f}"),
        TextColumn("gap={task.fields[gap]:.5f}"),
        TextColumn("raw={task.fields[raw_drift]:.2f}"),
        TextColumn("norm={task.fields[norm_drift]:.2f}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    def compute_feature_set_loss(
        fake_feat_set: Sequence[Tensor],
        real_feat_set: Sequence[Tensor],
        prefix: str,
    ) -> tuple[Tensor, float, float, float]:
        if len(fake_feat_set) != len(real_feat_set):
            raise RuntimeError(
                f"Feature extractor mismatch: {len(fake_feat_set)} fake branches vs "
                f"{len(real_feat_set)} real branches."
            )
        loss_sum: Tensor | None = None
        raw_sum = 0.0
        norm_sum = 0.0
        gap_sum = 0.0
        for branch_idx, (fake_feat, real_feat) in enumerate(zip(fake_feat_set, real_feat_set)):
            loss_per_sample, raw_drift_sq, norm_drift_sq = drifting_loss_from_features(
                fake_feat,
                real_feat,
                args.temperatures,
                norm_mode=args.drift_norm_mode,
                ema_state=drift_ema_state,
                ema_prefix=f"{prefix}/{branch_idx}",
                ema_decay=args.drift_ema_decay,
                eps=args.eps,
            )
            branch_loss = loss_per_sample.mean()
            loss_sum = branch_loss if loss_sum is None else loss_sum + branch_loss
            raw_sum += raw_drift_sq
            norm_sum += norm_drift_sq
            gap_sum += feature_moment_gap(fake_feat.detach(), real_feat.detach())

        if loss_sum is None:
            raise RuntimeError("No feature branches available for loss computation.")
        branch_count = float(max(1, len(fake_feat_set)))
        return (
            loss_sum / branch_count,
            raw_sum / branch_count,
            norm_sum / branch_count,
            gap_sum / branch_count,
        )

    generator.train()
    last_epoch = start_epoch - 1
    with progress:
        for epoch in range(start_epoch, args.epochs + 1):
            last_epoch = epoch
            task = progress.add_task(
                f"Epoch {epoch}/{args.epochs}",
                total=steps_per_epoch,
                loss=0.0,
                gap=0.0,
                raw_drift=0.0,
                norm_drift=0.0,
            )

            running_loss = 0.0
            running_gap = 0.0
            running_raw_drift = 0.0
            running_norm_drift = 0.0
            seen = 0

            for i, batch in enumerate(loader, start=1):
                if i > steps_per_epoch:
                    break
                if max_steps is not None and global_step >= max_steps:
                    break

                real, labels = unpack_batch(batch)
                real = real.to(device, non_blocking=(device.type == "cuda"))
                labels_t = None
                if cond_num_classes > 0:
                    if labels is None:
                        raise RuntimeError(
                            "class conditioning is enabled but this dataset does not provide labels."
                        )
                    labels_t = labels.to(device, dtype=torch.long, non_blocking=(device.type == "cuda"))
                z = torch.randn(real.shape[0], args.latent_dim, device=device)
                fake = generator(z, labels=labels_t)

                fake_feats = as_feature_list(feature_extractor(fake))
                with torch.no_grad():
                    real_feats = as_feature_list(feature_extractor(real))

                pixel_fake_feat = None
                pixel_real_feat = None
                if args.pixel_drift_weight > 0:
                    pixel_fake_feat = pixel_feature_extractor(fake)
                    with torch.no_grad():
                        pixel_real_feat = pixel_feature_extractor(real)

                # For class-conditional training, compute drifting losses within each class.
                if cond_num_classes > 0 and labels_t is not None:
                    loss = fake.new_zeros(())
                    total_raw_drift_sq = 0.0
                    total_norm_drift_sq = 0.0
                    total_gap = 0.0
                    weight_sum = 0.0

                    for cls in labels_t.unique(sorted=False):
                        cls_mask = labels_t == cls
                        cls_count = int(cls_mask.sum().item())
                        if cls_count < 2:
                            continue
                        w = float(cls_count) / float(real.shape[0])
                        cls_tag = int(cls.item())

                        cls_fake_feats = [f[cls_mask] for f in fake_feats]
                        cls_real_feats = [f[cls_mask] for f in real_feats]
                        cls_loss, cls_raw, cls_norm, cls_gap = compute_feature_set_loss(
                            cls_fake_feats, cls_real_feats, prefix=f"feature/c{cls_tag}"
                        )
                        loss = loss + w * cls_loss
                        total_raw_drift_sq += w * cls_raw
                        total_norm_drift_sq += w * cls_norm
                        total_gap += w * cls_gap
                        weight_sum += w

                        if args.pixel_drift_weight > 0:
                            assert pixel_fake_feat is not None and pixel_real_feat is not None
                            (
                                pixel_loss_per_sample,
                                pixel_raw_drift_sq,
                                pixel_norm_drift_sq,
                            ) = drifting_loss_from_features(
                                pixel_fake_feat[cls_mask],
                                pixel_real_feat[cls_mask],
                                args.temperatures,
                                norm_mode=args.drift_norm_mode,
                                ema_state=drift_ema_state,
                                ema_prefix=f"pixel/c{cls_tag}",
                                ema_decay=args.drift_ema_decay,
                                eps=args.eps,
                            )
                            loss = loss + args.pixel_drift_weight * w * pixel_loss_per_sample.mean()
                            total_raw_drift_sq += args.pixel_drift_weight * w * pixel_raw_drift_sq
                            total_norm_drift_sq += args.pixel_drift_weight * w * pixel_norm_drift_sq
                            total_gap += args.pixel_drift_weight * w * feature_moment_gap(
                                pixel_fake_feat[cls_mask].detach(),
                                pixel_real_feat[cls_mask].detach(),
                            )

                    if weight_sum > 0.0:
                        loss = loss / weight_sum
                        total_raw_drift_sq /= weight_sum
                        total_norm_drift_sq /= weight_sum
                        total_gap /= weight_sum
                    else:
                        loss, total_raw_drift_sq, total_norm_drift_sq, total_gap = (
                            compute_feature_set_loss(fake_feats, real_feats, prefix="feature/global")
                        )
                else:
                    loss, total_raw_drift_sq, total_norm_drift_sq, total_gap = (
                        compute_feature_set_loss(fake_feats, real_feats, prefix="feature/global")
                    )
                    if args.pixel_drift_weight > 0:
                        assert pixel_fake_feat is not None and pixel_real_feat is not None
                        (
                            pixel_loss_per_sample,
                            pixel_raw_drift_sq,
                            pixel_norm_drift_sq,
                        ) = drifting_loss_from_features(
                            pixel_fake_feat,
                            pixel_real_feat,
                            args.temperatures,
                            norm_mode=args.drift_norm_mode,
                            ema_state=drift_ema_state,
                            ema_prefix="pixel/global",
                            ema_decay=args.drift_ema_decay,
                            eps=args.eps,
                        )
                        loss = loss + args.pixel_drift_weight * pixel_loss_per_sample.mean()
                        total_raw_drift_sq += args.pixel_drift_weight * pixel_raw_drift_sq
                        total_norm_drift_sq += args.pixel_drift_weight * pixel_norm_drift_sq
                        total_gap += args.pixel_drift_weight * feature_moment_gap(
                            pixel_fake_feat.detach(), pixel_real_feat.detach()
                        )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(generator.parameters(), args.grad_clip)
                optimizer.step()

                global_step += 1
                seen += 1
                running_loss += loss.item()
                running_gap += total_gap
                running_raw_drift += total_raw_drift_sq
                running_norm_drift += total_norm_drift_sq

                avg_loss = running_loss / seen
                avg_gap = running_gap / seen
                avg_raw_drift = running_raw_drift / seen
                avg_norm_drift = running_norm_drift / seen
                progress.update(
                    task,
                    advance=1,
                    loss=avg_loss,
                    gap=avg_gap,
                    raw_drift=avg_raw_drift,
                    norm_drift=avg_norm_drift,
                )

                if args.log_every > 0 and global_step % args.log_every == 0:
                    LOGGER.info(
                        "step=%d loss=%.6f gap=%.6f raw_drift=%.6f norm_drift=%.6f",
                        global_step,
                        avg_loss,
                        avg_gap,
                        avg_raw_drift,
                        avg_norm_drift,
                    )

                if args.sample_every > 0 and global_step % args.sample_every == 0:
                    sample_path = samples_dir / f"step_{global_step:07d}.png"
                    save_samples(
                        generator,
                        sample_path,
                        args.sample_count,
                        args.sample_nrow,
                        args.latent_dim,
                        device,
                        fixed_z=fixed_sample_z,
                        fixed_labels=fixed_sample_labels,
                    )
                    LOGGER.info("Saved samples to %s", sample_path)

                if args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0:
                    ckpt_path = checkpoints_dir / f"step_{global_step:07d}.pt"
                    save_checkpoint(generator, optimizer, ckpt_path, args, global_step, epoch)
                    LOGGER.info("Saved checkpoint to %s", ckpt_path)

            progress.remove_task(task)

            if seen > 0:
                LOGGER.info(
                    "epoch=%d/%d avg_loss=%.6f avg_gap=%.6f avg_raw_drift=%.6f avg_norm_drift=%.6f",
                    epoch,
                    args.epochs,
                    running_loss / seen,
                    running_gap / seen,
                    running_raw_drift / seen,
                    running_norm_drift / seen,
                )

            if max_steps is not None and global_step >= max_steps:
                LOGGER.info("Reached max_steps=%d; stopping.", max_steps)
                break

    final_ckpt = checkpoints_dir / "final.pt"
    final_samples = samples_dir / "final.png"
    save_checkpoint(generator, optimizer, final_ckpt, args, global_step, last_epoch)
    save_samples(
        generator,
        final_samples,
        args.sample_count,
        args.sample_nrow,
        args.latent_dim,
        device,
        fixed_z=fixed_sample_z,
        fixed_labels=fixed_sample_labels,
    )
    LOGGER.info("Final checkpoint: %s", final_ckpt)
    LOGGER.info("Final samples: %s", final_samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drifting image generator training script.")

    parser.add_argument("--dataset", choices=["fashionmnist", "imagenet"], default="fashionmnist")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/drifting")

    parser.add_argument("--image-size", type=int, default=0, help="0 means auto (32 for FashionMNIST, 64 for ImageNet).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="0 means full dataloader.")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means no limit.")

    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=64)

    parser.add_argument(
        "--feature-backbone",
        choices=["auto", "pyramid", "resnet18"],
        default="auto",
        help="Feature extractor used to compute drift loss.",
    )
    parser.add_argument(
        "--pretrained-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pretrained weights when available (for resnet18).",
    )
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.05])
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--drift-norm-mode",
        choices=["batch", "ema"],
        default="ema",
        help="How to normalize feature drift magnitudes.",
    )
    parser.add_argument(
        "--drift-ema-decay",
        type=float,
        default=0.99,
        help="EMA decay used when --drift-norm-mode=ema.",
    )
    parser.add_argument(
        "--pixel-drift-weight",
        type=float,
        default=-1.0,
        help="Weight for additional pixel-space drift loss. "
        "If <0, auto-selects 0.5 for FashionMNIST and 0.0 for ImageNet.",
    )
    parser.add_argument(
        "--pixel-drift-size",
        type=int,
        default=16,
        help="Spatial size used for pixel-space drift features.",
    )

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-conditioning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable label-conditioned generation when labels are available.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Seed for sample snapshots; defaults to --seed.",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")

    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--sample-nrow", type=int, default=8)

    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint.")

    args = parser.parse_args()
    if args.image_size <= 0:
        args.image_size = 32 if args.dataset == "fashionmnist" else 64
    if args.sample_seed is None:
        args.sample_seed = args.seed
    if args.class_conditioning is None:
        args.class_conditioning = args.dataset == "imagenet"
    if not (0.0 < args.drift_ema_decay < 1.0):
        raise ValueError("--drift-ema-decay must be in (0, 1).")
    if args.pixel_drift_weight < 0:
        args.pixel_drift_weight = 0.5 if args.dataset == "fashionmnist" else 0.1
    return args


if __name__ == "__main__":
    train(parse_args())
