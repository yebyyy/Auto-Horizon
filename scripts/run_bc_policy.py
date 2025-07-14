"""
Real-time Behavioural-Cloning rollout for Forza Horizon 4.

Prereqs
-------
* FH4 in full-screen or borderless window.
* `scripts.capture` working (dxcam).
* Virtual gamepad driver + `vgamepad` installed.
* Trained checkpoint (.pt) from train_bc.

Usage
-----
$ conda activate fh4
$ python -m scripts.run_bc_policy --ckpt checkpoints/bc_e8_val0.026.pt
"""

from __future__ import annotations
import time, argparse, numpy as np, torch
import keyboard as kb
from scripts.capture import get_frame_stack
from policies.bc_policy import BCPolicy
from envs.actions import do_action

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/bc_e2_val0.026.pt", help="Path to the trained BC model checkpoint.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the policy.")
    parser.add_argument("--device", default="cuda", help="cuda | cpu")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    policy = BCPolicy(args.ckpt, device=device)
    period = 1.0 / args.fps

    print(f"BC rollout running at {args.fps} FPS on {device}. Press 'esc' to exit.")

    while True:
        if kb.is_pressed("esc"):
            print("Cancel before start")
            return
        if kb.is_pressed("space"):
            print("Starting rollout...")
            break
        time.sleep(0.05) # wait for space to be pressed

    action_id = 0
    try: 
        while not kb.is_pressed("esc"):
            tic = time.perf_counter()
            obs_stack = torch.from_numpy(get_frame_stack().numpy()).float()  # [S,C,H,W]
            action = policy.predict(obs_stack).numpy().tolist()  # [S,A]
            do_action(action)
            action_id += 1
            print(f"Action ID: {action_id}, Action: {action}")
            df = time.perf_counter() - tic
            if df < period:
                time.sleep(period - df)
    finally:
        print(f"Final action ID: {action_id}")
        do_action(([0.0, 0.0, 0.0]))  # release everything
        print("Released virtual pad. Exiting BC rollout.")

if __name__ == "__main__":
    main()
