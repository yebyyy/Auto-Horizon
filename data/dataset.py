"""
Dataset utilities for FH4 Behavioral Cloning
Given one or more **.npz** demo files recorded with scripts/collect_demos.py
this module exposes two helpers:

load_demos(files: list[str]) returns a tuple (obs, act)
FH4DemoDataset is the PyTorch Dataset wrapping those arrays and supporting an optional train/val split by indices.
"""

from __future__ import annotations

import os
from typing import List, Tuple
from scripts.capture import GRAY
import numpy as np
import torch
from torch.utils.data import Dataset

STACK_SIZE: int = 4                 # S
CHANNELS: int   = 1 if GRAY else 3  # C (RGB)
RESOLUTION: int = 84                # H, W

def load_demos(files: List[str] | str) -> Tuple[np.ndarray, np.ndarray]:
    # files is a list of file names
    if isinstance(files, str):
        files = ["./demos/" + files] if not files.startswith("./demos/") else [files]
        if not files:
            raise FileNotFoundError(f"No files match pattern: {files}")

    obs_list, act_list = [], []
    for f in files:
        data = np.load(f)
        obs_list.append(data["obs"].astype(np.float32))
        act_list.append(data["act"].astype(np.float32))

    obs = np.concatenate(obs_list, axis=0)
    act = np.concatenate(act_list, axis=0)

    # Sanity‑check shapes
    S, C, H, W = obs.shape[1:]
    assert S == STACK_SIZE and C == CHANNELS and H == RESOLUTION and W == RESOLUTION, f"Mismatch: expected ({STACK_SIZE},{CHANNELS},{RESOLUTION},{RESOLUTION}), got ({S},{C},{H},{W})"

    return obs, act

class FH4DemoDataset(Dataset):

    def __init__(self, obs: np.ndarray, act: np.ndarray, indices: List[int] | None = None):
        assert len(obs) == len(act), "obs and act length mismatch"
        self.obs = obs
        self.act = act
        self.indices = indices if indices is not None else list(range(len(obs)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (
            torch.from_numpy(self.obs[i]),  # (S,C,H,W)
            torch.from_numpy(self.act[i])   # (3,)
        )
