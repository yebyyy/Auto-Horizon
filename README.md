# Auto-Horizon
A self-driving Forza Horizon 4 agent

python -m scripts.train_bc --train_pattern "data/demos/*.npz" --epochs 16 --batch 128 --loader_workers 0
tensorboard --logdir runs

python -m scripts.eval_bc --demo "data/demos/*.npz" --run_id 8