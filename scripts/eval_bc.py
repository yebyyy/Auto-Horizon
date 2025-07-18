"""
Offline evaluation of the BC policy on the held-out sprint.

Usage:
$ conda activate fh4
$ python scripts/eval_bc.py \
        --demo data/demos/*.npz --run_id 8 \
        --ckpt checkpoints/bc_e1_val0.027.pt
"""

import argparse, glob, numpy as np, torch, torch.nn as nn, datetime
from data.dataset import load_demos
from scripts.train_bc import ConvPolicy, ResNetPolicy
import csv, pathlib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", default="data/demos/*.npz",
                        help="Glob pattern for demo files")
    parser.add_argument("--run_id", type=int, default=8, help="Run ID for eval, 1-indexed")
    parser.add_argument("--ckpt", default="checkpoints/bc_e15_val0.058.pt")
    parser.add_argument("--model", choices=["conv", "resnet"], default="conv",
                        help="Model architecture: conv | resnet")
    parser.add_argument("--csv",  default="eval_metrics.csv",
                    help="where to append results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data
    obs, act, run_lens = load_demos(args.demo)
    offsets = [0]
    for l in run_lens:
        offsets.append(offsets[-1] + l)
    start, end = offsets[args.run_id - 1], offsets[args.run_id]
    obs_test = torch.from_numpy(obs[start:end]).to(device)  # (N, S, C, H, W)
    act_test = torch.from_numpy(act[start:end]).to(device)  # (N, 3)

    print(f"using {end-start} frames from run {args.run_id} for evaluation")

    # load model
    if args.model == "resnet":
        model = ResNetPolicy(obs_test.shape).to(device)
    else:
        model = ConvPolicy(obs_test.shape).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()  # set to eval mode

    criterion = nn.SmoothL1Loss(reduction="mean")
    with torch.no_grad():
        pred = model(obs_test)  # (N, 3)
        loss = criterion(pred, act_test)
        mae = (pred - act_test).abs().mean(dim=0).cpu().numpy()

    print(f"Smooth L1 Loss: {loss.item():.4f}")
    print(f"MAE: steer {mae[0]:.4f}, throttle {mae[1]:.4f}, brake {mae[2]:.4f}")

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_frames = end - start
    row = [args.ckpt, args.run_id, n_frames, loss,
           mae[0], mae[1], mae[2], ts]

    header = ["ckpt","run_id","frames","smooth_l1",
              "mae_steer","mae_gas","mae_brake","timestamp"]

    write_header = not (csv_path := args.csv) or not glob.os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header: w.writerow(header)
        w.writerow(row)
    print(f"Appended results to {csv_path}")

if __name__ == "__main__":
    main()