# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 7, p: float = 0.1):
        super().__init__()
        pad = k // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.res(x)


class SimplePPickerNet(nn.Module):
    """
    输入: [B, C, T]
    输出: [B, T] logits
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 32, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.b1 = ConvBlock1D(base_channels, base_channels, k=7, p=dropout)
        self.b2 = ConvBlock1D(base_channels, base_channels * 2, k=7, p=dropout)
        self.b3 = ConvBlock1D(base_channels * 2, base_channels * 2, k=11, p=dropout)
        self.b4 = ConvBlock1D(base_channels * 2, base_channels, k=11, p=dropout)
        self.head = nn.Conv1d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.head(x)   # [B, 1, T]
        x = x.squeeze(1)   # [B, T]
        return x