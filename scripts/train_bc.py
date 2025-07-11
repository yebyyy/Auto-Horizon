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

