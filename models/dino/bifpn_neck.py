from __future__ import annotations

from typing import List

import torch
from torch import nn
import torch.nn.functional as F


class _SepConvGN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.gn = nn.GroupNorm(32, channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.pw(self.dw(x))))


class BiFPNLayer(nn.Module):
    """
    Lightweight BiFPN over same-channel feature list.
    Expects features ordered from high-resolution to low-resolution (e.g. P2->P5->P6).
    """

    def __init__(self, channels: int, num_levels: int):
        super().__init__()
        assert num_levels >= 3
        self.num_levels = int(num_levels)
        self.eps = 1e-4

        self.w_td = nn.ParameterList([nn.Parameter(torch.ones(2)) for _ in range(self.num_levels - 1)])
        self.w_bu = nn.ParameterList([nn.Parameter(torch.ones(2)) for _ in range(self.num_levels - 1)])

        self.conv_td = nn.ModuleList([_SepConvGN(channels) for _ in range(self.num_levels - 1)])
        self.conv_bu = nn.ModuleList([_SepConvGN(channels) for _ in range(self.num_levels - 1)])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(feats) == self.num_levels

        # top-down: from low-res -> high-res
        td = [None] * self.num_levels
        td[-1] = feats[-1]
        for i in range(self.num_levels - 2, -1, -1):
            w = F.relu(self.w_td[i])
            w = w / (w.sum() + self.eps)
            up = F.interpolate(td[i + 1], size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            td[i] = self.conv_td[i](w[0] * feats[i] + w[1] * up)

        # bottom-up: from high-res -> low-res
        out = [None] * self.num_levels
        out[0] = td[0]
        for i in range(1, self.num_levels):
            w = F.relu(self.w_bu[i - 1])
            w = w / (w.sum() + self.eps)
            down = F.max_pool2d(out[i - 1], kernel_size=2, stride=2)
            if down.shape[-2:] != td[i].shape[-2:]:
                down = F.interpolate(down, size=td[i].shape[-2:], mode="bilinear", align_corners=False)
            out[i] = self.conv_bu[i - 1](w[0] * td[i] + w[1] * down)
        return out


class BiFPNNeck(nn.Module):
    """
    Stackable BiFPN neck. Works on same-channel tensors.
    """

    def __init__(self, channels: int, num_levels: int, repeats: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([BiFPNLayer(channels, num_levels) for _ in range(int(repeats))])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        for layer in self.layers:
            feats = layer(feats)
        return feats

