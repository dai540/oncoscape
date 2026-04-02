from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _maybe_build_timm_encoder(name: str, pretrained: bool) -> tuple[nn.Module, int] | None:
    try:
        import timm  # type: ignore
    except Exception:
        return None
    try:
        encoder = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
        out_dim = int(getattr(encoder, "num_features", 0) or 0)
        if out_dim <= 0:
            return None
        return encoder, out_dim
    except Exception:
        return None


class SimpleCNNEncoder(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        return self.proj(h)


class GraphBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.msg = nn.Linear(hidden_dim, hidden_dim)
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return x
        src, dst = edge_index
        messages = self.msg(x[src])
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, messages)
        degree = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        agg = agg / degree.clamp_min(1.0).unsqueeze(1)
        out = self.self_proj(x) + agg
        out = self.norm(x + self.dropout(self.act(out)))
        return out


class DeepSpatialMultiTaskModel(nn.Module):
    def __init__(self, spec: dict[str, Any]) -> None:
        super().__init__()
        encoder_name = spec["encoder_name"]
        pretrained = bool(spec["encoder_pretrained"])
        encoder_out_dim = int(spec["encoder_out_dim"])
        built = None if encoder_name == "simple_cnn" else _maybe_build_timm_encoder(encoder_name, pretrained)
        if built is None:
            self.encoder = SimpleCNNEncoder(encoder_out_dim)
        else:
            backbone, backbone_dim = built
            self.encoder = nn.Sequential(backbone, nn.Linear(backbone_dim, encoder_out_dim))

        hidden_dim = int(spec["hidden_dim"])
        self.projector = nn.Sequential(
            nn.Linear(encoder_out_dim + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(float(spec["dropout"])),
        )
        self.graph_blocks = nn.ModuleList(
            [GraphBlock(hidden_dim, float(spec["dropout"])) for _ in range(int(spec["spatial_num_layers"]))]
        )
        self.compartment_head = nn.Linear(hidden_dim, int(spec["num_compartments"]))
        self.composition_head = nn.Linear(hidden_dim, int(spec["num_composition"]))
        self.program_head = nn.Linear(hidden_dim, int(spec["num_programs"]))

    def forward(self, patches: torch.Tensor, coords: torch.Tensor, edge_index: torch.Tensor) -> dict[str, torch.Tensor]:
        image_features = self.encoder(patches)
        coord_features = coords / 10000.0
        h = self.projector(torch.cat([image_features, coord_features], dim=1))
        for block in self.graph_blocks:
            h = block(h, edge_index)
        return {
            "latent": h,
            "compartment_logits": self.compartment_head(h),
            "composition_logits": self.composition_head(h),
            "program_values": self.program_head(h),
        }
