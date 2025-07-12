"""
Lightweight wrapper that loads a BC checkpoint and returns neural-network
actions given a stacked observation [S,C,H,W] (or batched).
"""

from __future__ import annotations
import torch
from scripts.train_bc import ConvPolicy

class BCPolicy:
    
    def __init__(self, 
                 ckpt_path: str,
                 in_shape: tuple[int, int, int, int, int] = (1, 4, 3, 84, 84),
                 device: str | torch.device = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.net = ConvPolicy(in_shape=in_shape).to(self.device)
        self.net.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.net.eval()

        for param in self.net.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def predict(self, obs_stack: torch.Tensor) -> torch.Tensor:
        if obs_stack.ndim == 4:
            obs_stack = obs_stack.unsqueeze(0)  # (S,C,H,W) -> (1,S,C,H,W), add batch dimension
            squeeze = True
        elif obs_stack.ndim == 5:
            squeeze = False
        else:
            raise ValueError(f"Invalid observation shape: {obs_stack.shape}")
        obs_stack = obs_stack.to(self.device)
        act = self.net(obs_stack)
        return act.squeeze(0).cpu() if squeeze else act.cpu()  # (1,S,A) -> (S,A)
    

if __name__ == "__main__":
    policy = BCPolicy(ckpt_path="checkpoints/bc_e5_val0.025.pt", in_shape=(1, 4, 3, 84, 84))
    dummy_obs = torch.randn(4, 3, 84, 84)
    print(policy.predict(dummy_obs))