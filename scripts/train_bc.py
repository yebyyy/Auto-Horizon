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
import os, argparse, time, datetime
import numpy as np
import torch, torch.nn as nn
from torchvision import models  # ResNet, VGG, etc.
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.dataset import load_demos, FH4DemoDataset, BalancedBatchSampler

def build_loaders(pattern, val_run, test_run, batch, workers):
    # Load training data
    obs, act, run_lens = load_demos(pattern)
    offsets = [0]
    for l in run_lens:
        offsets.append(offsets[-1] + l)

    def run_indices(run_id):
        return list(range(offsets[run_id-1], offsets[run_id]))

    idx_val = run_indices(val_run)
    idx_test = run_indices(test_run)
    idx_train = [i for i in range(len(obs)) if i not in idx_val and i not in idx_test]

    train_ds = FH4DemoDataset(obs, act, idx_train)
    val_ds = FH4DemoDataset(obs, act, idx_val)
    
    train_sampler = BalancedBatchSampler(act, idx_train, batch_size=batch, shuffle=True)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)

    print(f"Loaded {len(obs)} frames: train {len(train_ds)} | val {len(val_ds)} | test {len(idx_test)}")
    return train_loader, val_loader, train_ds, val_ds, obs.shape

class ConvPolicy(nn.Module):
    """Simple CNN policy for FH4 Behavioral Cloning."""
    def __init__(self, in_shape):
        super().__init__()
        in_channels = in_shape[1] * in_shape[2]
        # self.net is for observation input
        # (S,C,H,W) -> (batch, features)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=8, stride=4),  # downsample
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # inplace for memory efficiency
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Flatten(),  # flatten to (batch, features)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, in_shape[3], in_shape[4])
            flat_dim = self.net(dummy).shape[1]
        
        # self.head is for action output
        self.head = nn.Sequential(
            nn.Linear(flat_dim, 512),  # flat_dim because of Conv2d output
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 3),  # steer, gas, brake
            nn.Tanh()  # output range [-1, 1]
        )

    def forward(self, x):  # -> (B, 3)
        # x is (batch, S, C, H, W)
        b, s, c, h, w = x.shape
        x = x.view(b, s * c, h, w)  # reshape because Conv2d expects (batch, channels, height, width)
        out = self.head(self.net(x))  # (B, 3)
        steer = out[:, 0:1]
        gas = (out[:, 1:2] + 1) / 2  # scale to [0, 1]
        brake = (out[:, 2:3] + 1) / 2
        return torch.cat([steer, gas, brake], dim=1)
    
class ResNetPolicy(nn.Module):
    """ResNet-based policy"""
    def __init__(self, in_shape=(1, 4, 3, 84, 84), freeze_up_to=6):
        super().__init__()
        in_channels = in_shape[1] * in_shape[2]
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        layers = list(backbone.children())
        for l in layers[:freeze_up_to]:
            for p in l.parameters():
                p.requires_grad_(False)

        self.backbone = nn.Sequential(*layers[:-1])  # remove the final fc layer (nn.Linear(512, num_classes))
        self.head = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 3), nn.Tanh()  # steer, gas, brake
        )

        self.register_buffer("image_net_mean", torch.tensor([0.485, 0.456, 0.406]).repeat(4)[None, :, None, None])
        self.register_buffer("image_net_std", torch.tensor([0.229, 0.224, 0.225]).repeat(4)[None, :, None, None])

    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b, s * c, h, w)  # reshape to (batch, channels, height, width)
        x = (x - self.image_net_mean) / self.image_net_std  # normalize to ImageNet stats
        feats = self.backbone(x).flatten(1)
        out = self.head(feats)  # (B, 3)
        steer = out[:, 0:1]
        gas = (out[:, 1:2] + 1) / 2  # scale to [0, 1]
        brake = (out[:, 2:3] + 1) / 2
        return torch.cat([steer, gas, brake], dim=1)

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pattern", default="data/demos/*.npz",
                    help="Glob pattern for training demo files")
    parser.add_argument("--val_run", type=int, default=7, help="Run ID for validation set, 1-indexed")
    parser.add_argument("--test_run", type=int, default=8, help="Run ID for test set, 1-indexed")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--loader_workers", type=int, default=0, help="Number of DataLoader workers")
    parser.add_argument("--model", choices=["conv", "resnet"], default="conv",
                        help="Model architecture: conv | resnet")
    args = parser.parse_args()
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ld, val_ld, train_ds, val_ds, in_shape = build_loaders(
        args.train_pattern, args.val_run, args.test_run, args.batch, args.loader_workers
    )

    if args.model == "conv":
        model = ConvPolicy(in_shape).to(device)
    else:
        model = ResNetPolicy(in_shape).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    log_dir = os.path.join("runs", datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val = float("inf")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()  # set model to training mode
        running_loss = 0.0
        weight = torch.tensor([3., 1., 2.], device=device)  # weights for steer, gas, brake
        for obs_batch, act_batch in train_ld:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)
            pred = model(obs_batch)
            per_element_loss = nn.functional.smooth_l1_loss(pred, act_batch, reduction="none")
            loss = (per_element_loss * weight).mean()  # weighted loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(obs_batch)
        train_loss = running_loss / len(train_ds)
        # running loss is accumulated over the entire epoch
        # training loss is averaged over the dataset size

        # validation loop
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for obs_batch, act_batch in val_ld:
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)
                pred = model(obs_batch)
                per_element_loss = nn.functional.smooth_l1_loss(pred, act_batch, reduction="none")
                loss = (per_element_loss * weight).mean()
                running_loss += loss.item() * len(obs_batch)
        val_loss = running_loss / len(val_ds)

        # write to TensorBoard
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        print(f"epoch {epoch: 02d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"bc_e{epoch}_val{val_loss:.3f}.pt"))
            print(f"New best validation loss {val_loss:.4f}, saved to {ckpt_dir}/bc_e{epoch}_val{val_loss:.3f}.pt")

    print(f"Training complete. Best validation loss: {best_val:.4f}")
    writer.close()
    print(f"TensorBoard logs saved to {log_dir}")


if __name__ == "__main__":
    main()