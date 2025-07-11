"""scripts/train_bc.py
Behavioral Cloning training script for Forza Horizon 4 agent.

Assumes you have pre recorded demo files in  data/demos/*.npz and the
helper dataset utilities in data/dataset.py.

-----
$ conda activate fh4
$ python scripts/train_bc.py \
        --train_pattern data/demos/*.npz \
        --val_run    7 \
        --test_run   8 \
        --epochs 10 --batch 128 --lr 3e-4

-------
* TensorBoard logs in runs/
* Best validation checkpoint in  checkpoints/bc_policy.pt
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset import load_demos, FH4DemoDataset, STACK_SIZE, CHANNELS, RESOLUTION

class ConvPolicy(nn.Module):
    """Simple CNN policy for FH4 Behavioral Cloning."""
    def __init__(self):
        super().__init__()
        in_channels = STACK_SIZE * CHANNELS
        # self.net is for observation input
        # (S,C,H,W) -> (batch, features)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, stride=4),  # downsample
            nn.ReLU(inplace=True),  # inplace for memory efficiency
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),  # flatten to (batch, features)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, RESOLUTION, RESOLUTION)
            flat_dim = self.net(dummy).shape[1]
        
        # self.head is for action output
        self.head = nn.Sequential(
            nn.Linear(flat_dim, 512),  # flat_dim because of Conv2d output
            nn.ReLU(inplace=True),
            nn.Linear(512, 3), # output 3 actions: steer, gas, brake
            nn.Tanh()  # output range [-1, 1]
        )

    def forward(self, x):  # -> (B, 3)
        # x is (batch, S, C, H, W)
        B, S, C, H, W = x.shape
        x = x.view(B, S * C, H, W)  # reshape because Conv2d expects (batch, channels, height, width)
        out = self.head(self.net(x))  # (B, 3)
        steer = out[:, 0]
        gas = (out[:, 1] + 1) / 2  # scale to [0, 1]
        brake = (out[:, 2] + 1) / 2
        return torch.stack([steer, gas, brake], dim=1)
    
# utils------------------------------------------------------------------------------------------------------
def build_dataloaders(train_idx: List[int], val_idx: List[int], obs: np.array, act: np.ndarray, batch_size: int):
    ds_train = FH4DemoDataset(obs, act, indices=train_idx)
    ds_val = FH4DemoDataset(obs, act, indices=val_idx)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, pin_memory=True)  # pin_memory for faster GPU transfer
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, pin_memory=True)
    return dl_train, dl_val

# main