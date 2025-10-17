import os
import json
import torch
import wandb
import argparse
import numpy as np
from pathlib import Path
from k2vae.train import train
from k2vae.dataset import CustomDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DFINE")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--state-dim", type=int, default=128, help="state dimension")
    parser.add_argument("--patch-size", type=int, default=4, help="patch size")
    parser.add_argument("--mlp-num-layers", type=int, default=2, help="number of hidden layers in encoder and decoder")
    parser.add_argument("--mlp-hidden-dim", type=int, default=64, help="hidden dimension of encoder and decoder")
    parser.add_argument("--mlp-dropout", type=float, default=0.1, help="dropout probability for encoder and decoder")
    parser.add_argument("--mlp-activation", type=str, default="relu", help="nonlinear activation of encoder and decoder")
    parser.add_argument("--obs-context-len", type=int, default=64, help="context length in original signal space")
    parser.add_argument("--obs-forecast-len", type=int, default=16, help="forecast length in original signal space")
    parser.add_argument("--int-factor", type=int, default=5, help="factor for transformer")
    parser.add_argument("--int-dropout", type=float, default=0.1, help="dropout probability for transformer")
    parser.add_argument("--int-hidden-dim", type=int, default=64, help="hidden dimension of transformer model")
    parser.add_argument("--int-num-heads", type=int, default=4, help="number of heads in transformer")
    parser.add_argument("--int-dff", type=int, default=256, help="dff of transformer")
    parser.add_argument("--int-activation", type=str, default="relu", help="activation function of transformer")
    parser.add_argument("--int-num-layers", type=int, default=3, help="number of transformer encoder blocks")
    parser.add_argument("--kf-init", type=str, default="identity", help="method for initializing matrices in KF")
    parser.add_argument("--scaler-eps", type=float, default=1e-3, help="used for stability of division in scaler")
    parser.add_argument("--scaler-learnable", action="store_true", default=True, help="scaler has learnable weight and bias")
    parser.add_argument("--log-dir", type=str, default="log", help="logging directory")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--y-rec-weight", type=float, default=1.0, help="forecasting weight in loss")
    parser.add_argument("--x-rec-weight", type=float, default=1.0, help="context reconstruction weight in loss")
    parser.add_argument("--kl-beta", type=float, default=0.001, help="kl term weight in loss")
    parser.add_argument("--num-updates", type=int, default=2500, help="total number of updates")
    parser.add_argument("--test-interval", type=int, default=100, help="testing interval")
    parser.add_argument("--run-id", type=str, default="K2VAE", help="name of the run")
    parser.add_argument("--disable-gpu", action="store_true", default=False, help="disable using gpu")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0, help="clip gradients to this value")
    parser.add_argument("--notes", type=str, default="", help="extra notes to add to the run")
    parser.add_argument("--data-path", type=str, default="./datasets/", help="path to the data")

    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)
    
    # prepare logging
    save_dir = Path(args.log_dir) / args.run_id
    os.makedirs(save_dir, exist_ok=True)
    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f)
    
    wandb.init(
        project="Time Series Forecasting",
        name="K2VAE",
        config=vars(args),
        notes=args.notes,
    )

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # load datasets
    dataset = CustomDataset(path=Path(args.data_path))

    model = train(
        args=args,
        train_dataset=dataset,
        test_dataset=dataset,
    )

    wandb.finish()
