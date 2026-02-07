#!/usr/bin/env python3
"""
Drifting Models for Image Generation
====================================

A practical image-generation script following the core training algorithm from
"Generative Modeling via Drifting" (arXiv:2602.04770) and the local
`drifting_ref.py` reference:

  loss = || f(eps) - stopgrad(f(eps) + V) ||^2

where V is computed with the doubly-normalized kernelized drifting field.

Supported datasets:
- FashionMNIST (auto-download)
- ImageNet-style folder (ImageFolder at <data-root>/train/*)
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
    import torchinfo
except Exception:  # pragma: no cover
    torchinfo = None

try:
    from torchvision.models import ResNet18_Weights
except Exception:  # pragma: no cover
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
    mask_self_negatives: bool = False,
    eps: float = 1e-12,
) -> Tensor:
    """
    Algorithm 2 style compute_V with doubly-normalized affinities.
    """
    n, n_pos = x.shape[0], y_pos.shape[0]

    dist_pos = torch.cdist(x, y_pos)
    dist_neg = torch.cdist(x, y_neg)

    if mask_self_negatives:
        if n != y_neg.shape[0]:
            raise ValueError("mask_self_negatives=True requires x and y_neg to have same batch size.")
        dist_neg = dist_neg + torch.eye(n, device=x.device, dtype=x.dtype) * 1e6

    logit = torch.cat([-dist_pos / temp, -dist_neg / temp], dim=1)
    a_row = logit.softmax(dim=-1)
    a_col = logit.softmax(dim=-2)
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
    Appendix feature normalization: average pairwise distance ~ sqrt(C).
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
    eps: float = 1e-6,
) -> tuple[Tensor, float, float]:
    """
    Compute normalized drift in feature space with multi-temperature aggregation.
    """
    x_norm, pos_norm, neg_norm = normalize_feature_space(x_feat, pos_feat, x_feat, eps=eps)
    c = x_feat.shape[-1]
    temp_scale = math.sqrt(float(c))

    total = torch.zeros_like(x_norm)
    raw_vals: list[float] = []
    norm_vals: list[float] = []

    for tau in temperatures:
        tau_eff = max(float(tau), eps) * temp_scale
        v = compute_drift(
            x_norm,
            pos_norm,
            neg_norm,
            temp=tau_eff,
            mask_self_negatives=True,
            eps=eps,
        )

        with torch.no_grad():
            raw_sq = v.pow(2).sum(dim=-1).mean()
            lam_sq = (raw_sq / float(c)).clamp_min(eps)
            lam = torch.sqrt(lam_sq)

        v_norm = v / lam
        total = total + v_norm

        with torch.no_grad():
            raw_vals.append(raw_sq.item())
            norm_vals.append(v_norm.pow(2).sum(dim=-1).mean().item())

    raw_drift_sq = sum(raw_vals) / max(1, len(raw_vals))
    norm_drift_sq = sum(norm_vals) / max(1, len(norm_vals))
    return total, raw_drift_sq, norm_drift_sq


def drifting_loss_from_features(
    fake_feat: Tensor,
    real_feat: Tensor,
    temperatures: Sequence[float],
    eps: float = 1e-6,
) -> tuple[Tensor, float, float]:
    """
    stopgrad(fake + V) target as in Algorithm 1.
    """
    with torch.no_grad():
        drift, raw_drift_sq, norm_drift_sq = compute_feature_drift(
            fake_feat.detach(),
            real_feat.detach(),
            temperatures,
            eps=eps,
        )
        target = (fake_feat + drift).detach()

    loss_per_sample = (fake_feat - target).pow(2).mean(dim=-1)
    return loss_per_sample, raw_drift_sq, norm_drift_sq


def as_feature_list(feats: Tensor | Sequence[Tensor]) -> list[Tensor]:
    if torch.is_tensor(feats):
        return [feats]
    return list(feats)


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


def _sincos_1d(embed_dim: int, pos: Tensor) -> Tensor:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim for 1D sin-cos must be even.")
    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / float(embed_dim // 2)))
    out = pos.float()[:, None] * omega[None, :]
    return torch.cat([out.sin(), out.cos()], dim=1)


def build_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int) -> Tensor:
    if embed_dim % 4 != 0:
        raise ValueError("embed_dim must be divisible by 4 for 2D sin-cos embedding.")
    yy, xx = torch.meshgrid(
        torch.arange(grid_h, dtype=torch.float32),
        torch.arange(grid_w, dtype=torch.float32),
        indexing="ij",
    )
    pos_y = _sincos_1d(embed_dim // 2, yy.reshape(-1))
    pos_x = _sincos_1d(embed_dim // 2, xx.reshape(-1))
    pos = torch.cat([pos_y, pos_x], dim=1)
    return pos[None, :, :]


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        inner = int(hidden_dim * mlp_ratio)
        self.fc1 = nn.Linear(hidden_dim, inner)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(inner, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(hidden_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.ffn = FeedForward(hidden_dim, mlp_ratio=mlp_ratio)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))

        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = self.ada(c).chunk(
            6, dim=1
        )
        x = x + gate_attn[:, None, :] * self.attn(_modulate(self.norm1(x), shift_attn, scale_attn))
        x = x + gate_ffn[:, None, :] * self.ffn(_modulate(self.norm2(x), shift_ffn, scale_ffn))
        return x


class DiTFinalLayer(nn.Module):
    def __init__(self, hidden_dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 2))
        self.to_patch = nn.Linear(hidden_dim, patch_size * patch_size * out_channels)

        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)
        nn.init.zeros_(self.to_patch.weight)
        nn.init.zeros_(self.to_patch.bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.ada(c).chunk(2, dim=1)
        x = _modulate(self.norm(x), shift, scale)
        return self.to_patch(x)


class DiTGenerator(nn.Module):
    """
    DiT-style generator with adaLN-zero conditioning.
    """

    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        image_size: int,
        num_classes: int = 0,
        patch_size: int = 4,
        hidden_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.hidden_dim = hidden_dim
        self.num_classes = max(0, int(num_classes))

        self.noise_proj = nn.Linear(latent_dim, out_channels * image_size * image_size)
        self.patch_embed = nn.Conv2d(
            out_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        pos = build_2d_sincos_pos_embed(hidden_dim, self.grid_size, self.grid_size)
        self.register_buffer("pos_embed", pos, persistent=False)

        if self.num_classes > 0:
            self.null_label = self.num_classes
            self.class_embed = nn.Embedding(self.num_classes + 1, hidden_dim)
        else:
            self.null_label = -1
            self.class_embed = None
        self.uncond_embed = nn.Parameter(torch.zeros(1, hidden_dim))
        self.cond_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.final = DiTFinalLayer(hidden_dim, patch_size=patch_size, out_channels=out_channels)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_condition(self, labels: Tensor | None, batch: int, cond_drop_prob: float) -> Tensor:
        if self.class_embed is None:
            cond = self.uncond_embed.expand(batch, -1)
            return self.cond_mlp(cond)

        if labels is None:
            raise ValueError("labels are required when class conditioning is enabled.")
        if labels.shape[0] != batch:
            raise ValueError("labels batch size mismatch.")

        y = labels
        if self.training and cond_drop_prob > 0:
            keep = torch.rand(batch, device=labels.device) >= cond_drop_prob
            nulls = torch.full_like(labels, self.null_label)
            y = torch.where(keep, labels, nulls)
        cond = self.class_embed(y)
        return self.cond_mlp(cond)

    def _unpatchify(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        g = self.grid_size
        p = self.patch_size
        c = self.out_channels
        if n != g * g:
            raise RuntimeError("Unexpected token count in unpatchify.")
        x = x.view(b, g, g, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(b, c, g * p, g * p)

    def forward(self, z: Tensor, labels: Tensor | None = None, cond_drop_prob: float = 0.0) -> Tensor:
        b = z.shape[0]
        eps_img = self.noise_proj(z).view(b, self.out_channels, self.image_size, self.image_size)
        x = self.patch_embed(eps_img).flatten(2).transpose(1, 2)
        x = x + self.pos_embed.to(device=x.device, dtype=x.dtype)
        c = self._get_condition(labels=labels, batch=b, cond_drop_prob=cond_drop_prob)
        for block in self.blocks:
            x = block(x, c)
        x = self.final(x, c)
        x = self._unpatchify(x)
        return torch.tanh(x)

    @torch.no_grad()
    def generate(self, n: int, device: torch.device, labels: Tensor | None = None) -> Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        return self(z, labels=labels, cond_drop_prob=0.0)


class ConvGenerator(nn.Module):
    """
    Simple upsampling conv generator (kept intentionally minimal).
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
        self.class_embed = nn.Embedding(self.num_classes, latent_dim) if self.num_classes > 0 else None
        cond_dim = latent_dim + (latent_dim if self.num_classes > 0 else 0)

        self.start_channels = base_channels * min(8, 2**num_upsamples)
        self.project = nn.Linear(cond_dim, self.start_channels * 4 * 4)

        self.stages = nn.ModuleList()
        channels = self.start_channels
        current_size = 4
        while current_size < image_size:
            next_channels = max(base_channels, channels // 2)
            groups = 8 if next_channels % 8 == 0 else 1
            self.stages.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(channels, next_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(groups, next_channels),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(next_channels, next_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(groups, next_channels),
                    nn.SiLU(inplace=True),
                )
            )
            channels = next_channels
            current_size *= 2

        self.to_image = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, z: Tensor, labels: Tensor | None = None, cond_drop_prob: float = 0.0) -> Tensor:
        del cond_drop_prob
        if self.class_embed is not None:
            if labels is None:
                raise ValueError("labels are required when class conditioning is enabled.")
            cond = torch.cat([z, self.class_embed(labels)], dim=-1)
        else:
            cond = z

        h = self.project(cond).view(cond.shape[0], self.start_channels, 4, 4)
        for stage in self.stages:
            h = stage(h)
        return self.to_image(h)

    @torch.no_grad()
    def generate(self, n: int, device: torch.device, labels: Tensor | None = None) -> Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        return self(z, labels=labels)


class PyramidFeatureExtractor(nn.Module):
    """
    Lightweight multi-scale feature extractor for small/fast runs.
    Returns list of [B, T, C] features.
    """

    def __init__(self, pool_sizes: Sequence[int] = (8, 4, 2, 1)):
        super().__init__()
        self.pool_sizes = tuple(pool_sizes)

    def forward(self, x: Tensor) -> list[Tensor]:
        feats: list[Tensor] = []
        b = x.shape[0]
        for size in self.pool_sizes:
            pooled = F.adaptive_avg_pool2d(x, output_size=(size, size))
            feats.append(pooled.permute(0, 2, 3, 1).reshape(b, size * size, pooled.shape[1]))
        return feats


class ResNet18FeatureExtractor(nn.Module):
    """
    Frozen ResNet18 multi-scale feature encoder.
    Returns list of [B, T, C] features.
    """

    def __init__(self, pretrained: bool = True, pool_sizes: Sequence[int] = (8, 4, 2, 1)):
        super().__init__()
        self.pool_sizes = tuple(pool_sizes)

        weights = None
        if pretrained and ResNet18_Weights is not None:
            weights = ResNet18_Weights.DEFAULT

        try:
            model = resnet18(weights=weights)
        except Exception as exc:
            if pretrained:
                LOGGER.warning("Could not load pretrained ResNet18 (%s). Falling back to random init.", exc)
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

    def forward(self, x: Tensor) -> list[Tensor]:
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

        feats: list[Tensor] = []
        for f, size in zip((f1, f2, f3, f4), self.pool_sizes):
            pooled = F.adaptive_avg_pool2d(f, output_size=(size, size))
            b, c, h, w = pooled.shape
            feats.append(pooled.permute(0, 2, 3, 1).reshape(b, h * w, c))
        return feats


def build_feature_extractor(name: str, dataset_name: str, use_pretrained: bool) -> tuple[nn.Module, str]:
    resolved = name
    if name == "auto":
        resolved = "pyramid" if dataset_name == "fashionmnist" else "resnet18"

    if resolved == "pyramid":
        return PyramidFeatureExtractor(), resolved
    if resolved == "resnet18":
        return ResNet18FeatureExtractor(pretrained=use_pretrained), resolved
    raise ValueError(f"Unsupported feature extractor: {name}")


def build_generator(
    arch: str,
    latent_dim: int,
    out_channels: int,
    image_size: int,
    base_channels: int,
    num_classes: int,
    patch_size: int,
    hidden_dim: int,
    depth: int,
    num_heads: int,
    mlp_ratio: float,
) -> nn.Module:
    if arch == "conv":
        return ConvGenerator(
            latent_dim=latent_dim,
            out_channels=out_channels,
            image_size=image_size,
            base_channels=base_channels,
            num_classes=num_classes,
        )
    if arch == "dit":
        return DiTGenerator(
            latent_dim=latent_dim,
            out_channels=out_channels,
            image_size=image_size,
            num_classes=num_classes,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
    raise ValueError(f"Unsupported generator architecture: {arch}")


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
                    image_size,
                    scale=(0.6, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_dataset(
    dataset_name: str,
    data_root: Path,
    image_size: int,
) -> tuple[torch.utils.data.Dataset, int, int]:
    transform = image_transform(dataset_name, image_size)

    if dataset_name == "fashionmnist":
        ds = datasets.FashionMNIST(root=str(data_root), train=True, transform=transform, download=True)
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
                "ImageNet path not found. Use --data-root containing either "
                "'train/<class_name>/*' (ImageFolder) or official torchvision ImageNet metadata layout."
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
    generator: nn.Module,
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
    images = generator(z, labels=labels, cond_drop_prob=0.0).cpu()
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    save_image(grid, str(out_path))
    if was_training:
        generator.train()


def save_checkpoint(
    generator: nn.Module,
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
    table = Table(title="Drifting Image Training (Paper-Core)")
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
    table.add_row("generator_arch", args.generator_arch)
    if args.generator_arch == "conv":
        table.add_row("base_channels", str(args.base_channels))
    else:
        table.add_row("dit_patch_size", str(args.dit_patch_size))
        table.add_row("dit_hidden_dim", str(args.dit_hidden_dim))
        table.add_row("dit_depth", str(args.dit_depth))
        table.add_row("dit_heads", str(args.dit_heads))
        table.add_row("dit_mlp_ratio", f"{args.dit_mlp_ratio:g}")
        table.add_row("cond_drop_prob", f"{args.cond_drop_prob:g}")
    table.add_row("feature_extractor", feature_name)
    table.add_row("temperatures", ",".join(f"{t:g}" for t in args.temperatures))
    table.add_row("optimizer", "AdamW(beta1=0.9,beta2=0.95)")
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
        args.feature_backbone,
        args.dataset,
        args.pretrained_features,
    )
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    for p in feature_extractor.parameters():
        p.requires_grad_(False)

    generator = build_generator(
        arch=args.generator_arch,
        latent_dim=args.latent_dim,
        out_channels=out_channels,
        image_size=args.image_size,
        base_channels=args.base_channels,
        num_classes=cond_num_classes,
        patch_size=args.dit_patch_size,
        hidden_dim=args.dit_hidden_dim,
        depth=args.dit_depth,
        num_heads=args.dit_heads,
        mlp_ratio=args.dit_mlp_ratio,
    ).to(device)

    if torchinfo is not None:
        try:
            console.print(
                torchinfo.summary(
                    generator,
                    input_size=(1, args.latent_dim),
                    device=str(device),
                    verbose=0,
                )
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Could not render model summary: %s", exc)
        # Some torchinfo paths can leave modules on CPU; force device consistency.
        generator = generator.to(device)

    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    sample_noise_generator = torch.Generator(device="cpu")
    sample_noise_generator.manual_seed(args.sample_seed)
    fixed_sample_z = torch.randn(args.sample_count, args.latent_dim, generator=sample_noise_generator)
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

    render_config_table(console, args, len(dataset), num_classes, feature_name, device, run_dir)

    steps_per_epoch = len(loader) if args.steps_per_epoch <= 0 else min(args.steps_per_epoch, len(loader))
    max_steps = args.max_steps if args.max_steps > 0 else None

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TextColumn("loss={task.fields[loss]:.5f}"),
        TextColumn("raw={task.fields[raw_drift]:.3f}"),
        TextColumn("norm={task.fields[norm_drift]:.3f}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    def compute_feature_set_loss(
        fake_feat_set: Sequence[Tensor],
        real_feat_set: Sequence[Tensor],
    ) -> tuple[Tensor, float, float]:
        if len(fake_feat_set) != len(real_feat_set):
            raise RuntimeError(
                f"Feature extractor mismatch: {len(fake_feat_set)} fake branches vs "
                f"{len(real_feat_set)} real branches."
            )

        loss_sum: Tensor | None = None
        raw_sum = 0.0
        norm_sum = 0.0

        for fake_feat, real_feat in zip(fake_feat_set, real_feat_set):
            fake_flat = fake_feat.reshape(-1, fake_feat.shape[-1])
            real_flat = real_feat.reshape(-1, real_feat.shape[-1])

            loss_per_sample, raw_drift_sq, norm_drift_sq = drifting_loss_from_features(
                fake_flat,
                real_flat,
                args.temperatures,
                eps=args.eps,
            )
            branch_loss = loss_per_sample.mean()
            loss_sum = branch_loss if loss_sum is None else loss_sum + branch_loss
            raw_sum += raw_drift_sq
            norm_sum += norm_drift_sq

        if loss_sum is None:
            raise RuntimeError("No feature branches available for loss computation.")

        count = float(max(1, len(fake_feat_set)))
        return loss_sum / count, raw_sum / count, norm_sum / count

    generator.train()
    last_epoch = start_epoch - 1

    with progress:
        for epoch in range(start_epoch, args.epochs + 1):
            last_epoch = epoch
            task = progress.add_task(
                f"Epoch {epoch}/{args.epochs}",
                total=steps_per_epoch,
                loss=0.0,
                raw_drift=0.0,
                norm_drift=0.0,
            )

            running_loss = 0.0
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
                        raise RuntimeError("class conditioning enabled but dataset does not provide labels")
                    labels_t = labels.to(device, dtype=torch.long, non_blocking=(device.type == "cuda"))

                z = torch.randn(real.shape[0], args.latent_dim, device=device)
                fake = generator(
                    z,
                    labels=labels_t,
                    cond_drop_prob=(args.cond_drop_prob if cond_num_classes > 0 else 0.0),
                )

                fake_feats = as_feature_list(feature_extractor(fake))
                with torch.no_grad():
                    real_feats = as_feature_list(feature_extractor(real))

                if cond_num_classes > 0 and labels_t is not None:
                    loss = fake.new_zeros(())
                    total_raw = 0.0
                    total_norm = 0.0
                    weight_sum = 0.0

                    for cls in labels_t.unique(sorted=False):
                        cls_mask = labels_t == cls
                        cls_count = int(cls_mask.sum().item())
                        if cls_count < 2:
                            continue

                        w = float(cls_count) / float(real.shape[0])
                        cls_fake_feats = [f[cls_mask] for f in fake_feats]
                        cls_real_feats = [f[cls_mask] for f in real_feats]
                        cls_loss, cls_raw, cls_norm = compute_feature_set_loss(cls_fake_feats, cls_real_feats)

                        loss = loss + w * cls_loss
                        total_raw += w * cls_raw
                        total_norm += w * cls_norm
                        weight_sum += w

                    if weight_sum > 0.0:
                        loss = loss / weight_sum
                        total_raw /= weight_sum
                        total_norm /= weight_sum
                    else:
                        loss, total_raw, total_norm = compute_feature_set_loss(fake_feats, real_feats)
                else:
                    loss, total_raw, total_norm = compute_feature_set_loss(fake_feats, real_feats)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(generator.parameters(), args.grad_clip)
                optimizer.step()

                global_step += 1
                seen += 1
                running_loss += loss.item()
                running_raw_drift += total_raw
                running_norm_drift += total_norm

                avg_loss = running_loss / seen
                avg_raw = running_raw_drift / seen
                avg_norm = running_norm_drift / seen
                progress.update(task, advance=1, loss=avg_loss, raw_drift=avg_raw, norm_drift=avg_norm)

                if args.log_every > 0 and global_step % args.log_every == 0:
                    LOGGER.info(
                        "step=%d loss=%.6f raw_drift=%.6f norm_drift=%.6f",
                        global_step,
                        avg_loss,
                        avg_raw,
                        avg_norm,
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
                    "epoch=%d/%d avg_loss=%.6f avg_raw_drift=%.6f avg_norm_drift=%.6f",
                    epoch,
                    args.epochs,
                    running_loss / seen,
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
    parser = argparse.ArgumentParser(description="Drifting image generator training script (paper-core).")

    parser.add_argument("--dataset", choices=["fashionmnist", "imagenet"], default="fashionmnist")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/drifting")

    parser.add_argument(
        "--image-size",
        type=int,
        default=0,
        help="0 means auto (32 for FashionMNIST, 64 for ImageNet).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="0 means full dataloader.")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means no limit.")

    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument(
        "--generator-arch",
        choices=["conv", "dit"],
        default="dit",
        help="Generator backbone. 'dit' is a DiT-style transformer with adaLN-zero conditioning.",
    )
    parser.add_argument("--dit-patch-size", type=int, default=4)
    parser.add_argument("--dit-hidden-dim", type=int, default=768)
    parser.add_argument("--dit-depth", type=int, default=12)
    parser.add_argument("--dit-heads", type=int, default=12)
    parser.add_argument("--dit-mlp-ratio", type=float, default=4.0)

    parser.add_argument(
        "--feature-backbone",
        choices=["auto", "pyramid", "resnet18"],
        default="auto",
        help="Feature extractor used for drifting loss.",
    )
    parser.add_argument(
        "--pretrained-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pretrained weights when available (for resnet18).",
    )

    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.02, 0.05, 0.2])
    parser.add_argument("--eps", type=float, default=1e-6)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=2.0)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-conditioning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable label-conditioned generation when labels are available.",
    )
    parser.add_argument(
        "--cond-drop-prob",
        type=float,
        default=0.1,
        help="Classifier-free conditioning dropout probability (used when class conditioning is enabled).",
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
    if not (0.0 <= args.cond_drop_prob < 1.0):
        raise ValueError("--cond-drop-prob must be in [0, 1).")
    return args


if __name__ == "__main__":
    train(parse_args())
