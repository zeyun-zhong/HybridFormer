"""
Extracted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/swin_transformer_v2.py
and https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py
"""

import torch
import torch.nn as nn
import math
import numpy as np
from typing import Tuple


class DynamicRelativePositionBias(nn.Module):
    """
    2D case, for images or spatio-temporal skeleton
    """
    def __init__(
            self, num_heads: int, window_size: Tuple[int, int],
            pretrained_window_size: Tuple[int, int] = (0, 0),
            mlp_dim=512,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size

        # MLP to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, mlp_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, num_heads, bias=False)
        )

        # Compute relative_coords_table
        relative_coords_table = self.create_relative_coords_table()

        # Compute relative_position_index
        self.relative_position_index = self.create_relative_position_index()

        # Register relative_coords_table as a buffer
        self.register_buffer('relative_coords_table', relative_coords_table)

    def create_relative_coords_table(self):
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([
            relative_coords_h,
            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)

        if self.pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (self.pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)

        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / math.log2(8)
        return relative_coords_table

    def create_relative_position_index(self):
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index

    def forward(self):
        relative_position_bias_table = self.cpb_mlp(
            self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0,
                                                                1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        return relative_position_bias


class DynamicRelativePositionBias1D(nn.Module):
    """
    1D case, for temporal relative position bias
    """
    def __init__(
            self, num_heads: int, window_size: int,
            pretrained_window_size: int = 0, mlp_dim=512, num_points=25):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_points = num_points
        self.pretrained_window_size = pretrained_window_size

        # MLP to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(1, mlp_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, num_heads, bias=False)
        )

        # Compute relative_coords_table for 1D
        relative_coords_table = self.create_relative_coords_table_1d()

        # Compute relative_position_index for 1D
        self.relative_position_index = self.create_relative_position_index_1d()

        # Register relative_coords_table as a buffer
        self.register_buffer('relative_coords_table', relative_coords_table)

    def create_relative_coords_table_1d(self):
        relative_coords = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32).unsqueeze(0)

        if self.pretrained_window_size > 0:
            relative_coords /= (self.pretrained_window_size - 1)
        else:
            relative_coords /= (self.window_size - 1)

        relative_coords *= 8
        relative_coords = torch.sign(relative_coords) * torch.log2(torch.abs(relative_coords) + 1.0) / math.log2(8)

        # Reshape to match MLP input dimension: [num_pairs, 1]
        relative_coords = relative_coords.unsqueeze(-1)
        return relative_coords

    def create_relative_position_index_1d(self):
        coords = torch.arange(self.window_size)
        relative_coords = coords[:, None] - coords[None, :]
        relative_coords += self.window_size - 1
        relative_position_index = relative_coords
        return relative_position_index

    def forward(self):
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1)
        relative_position_bias = relative_position_bias.repeat_interleave(
            self.num_points, dim=0).repeat_interleave(self.num_points, dim=1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        return relative_position_bias


class HopRelativePositionBias(nn.Module):
    def __init__(self, num_points, A, num_heads=6, mlp_dim=512, num_frames=64,
                 hops=None):
        super().__init__()
        self.num_points = num_points
        self.num_frames = num_frames
        self.num_heads = num_heads
        self.A = A

        # MLP to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(1, mlp_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, num_heads, bias=False)
        )

        self.hops = torch.tensor(hops).long()
        relative_hop_table = self.create_relative_hop_table(self.hops.max())

        # Register relative_hop_table as a buffer
        self.register_buffer('relative_hop_table', relative_hop_table)

    def create_relative_hop_table(self, max_hop: int):
        relative_hops = torch.arange(0, max_hop + 1, dtype=torch.float32).unsqueeze(0)
        relative_hops /= max_hop
        relative_hops *= 8
        relative_hops = (relative_hops + 1.0) / math.log2(8)
        return relative_hops.unsqueeze(-1)

    def forward(self):
        relative_hop_bias_table = self.cpb_mlp(self.relative_hop_table).view(-1, self.num_heads)
        relative_hop_bias = relative_hop_bias_table[self.hops.view(-1)].view(
            self.num_points, self.num_points, -1)

        if self.num_frames is not None:
            relative_hop_bias = relative_hop_bias.repeat(self.num_frames, self.num_frames, 1)

        relative_hop_bias = relative_hop_bias.permute(2, 0, 1).contiguous()
        relative_hop_bias = 16 * torch.sigmoid(relative_hop_bias)
        return relative_hop_bias

