"""
Dataset utilities for FH4 Behavioral Cloning
Given one or more **.npz** demo files recorded with scripts/collect_demos.py
this module exposes two helpers:

load_demos(files: list[str]) returns a tuple (obs, act)
FH4DemoDataset is the PyTorch Dataset wrapping those arrays and supporting an optional train/val split by indices.
"""

from __future__ import annotations

import os, glob, itertools
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

GRAY = False
STACK_SIZE: int = 4                 # S
CHANNELS: int   = 1 if GRAY else 4  # C (RGB + mask)
RESOLUTION: int = 84                # H, W

def load_demos(files: List[str] | str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    # files is a list of file names
    if isinstance(files, str):
        files = sorted(glob.glob(files))  # glob is used to match patterns like "data/demos/*.npz"
        if not files:
            raise FileNotFoundError(f"No files match pattern: {files}")

    obs_list, act_list, run_lengths = [], [], []
    for f in files:
        data = np.load(f)
        obs_list.append(data["obs"].astype(np.float32))
        act_list.append(data["act"].astype(np.float32))
        run_lengths.append(len(data["obs"]))

    obs = np.concatenate(obs_list, axis=0)
    act = np.concatenate(act_list, axis=0)

    # Sanity‑check shapes
    S, C, H, W = obs.shape[1:]
    assert S == STACK_SIZE and C == CHANNELS and H == RESOLUTION and W == RESOLUTION, f"Mismatch: expected ({STACK_SIZE},{CHANNELS},{RESOLUTION},{RESOLUTION}), got ({S},{C},{H},{W})"

    return obs, act, run_lengths

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
    
class BalancedBatchSampler(Sampler):
    """
    Yields indices so that every batch contains a 50-50 mix of
    |steer| > 0.3 or brake > 0.4
    and common straight gas frames.
    """
    def __init__(self, actions: np.ndarray, indices: List[int], batch_size: int, shuffle: bool = True):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle

        steer_turn = np.abs(actions[:, 0]) > 0.3
        heavy_brake = actions[:, 2] > 0.4
        rare_mask = steer_turn | heavy_brake

        self.rare = [k for k, idx in enumerate(indices) if rare_mask[idx]]
        self.common = [k for k, idx in enumerate(indices) if not rare_mask[idx]]
        self.num_batches = len(indices) // batch_size

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.rare)
            np.random.shuffle(self.common)
        
        # itertools.cycle is used to ensure we can always fill a batch
        r_iter = itertools.cycle(self.rare)
        c_iter = itertools.cycle(self.common)

        half = self.batch_size // 2
        for _ in range(self.num_batches):
            batch = [next(r_iter) for _ in range(half)] + [next(c_iter) for _ in range(self.batch_size - half)]
            if self.shuffle:
                np.random.shuffle(batch)
            yield batch  # yield means we can iterate over this sampler

    def __len__(self):
        return self.num_batches