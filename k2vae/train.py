import wandb
import torch
from pathlib import Path
from tqdm import tqdm
from argparse import Namespace
from .dataset import CustomDataset
from .model import K2VAE
from torch.nn.utils import clip_grad_norm_


def train(
    args: Namespace,
    train_dataset: CustomDataset,
    test_dataset: CustomDataset
):
    
    # define models and optimizer
    device = "cuda" if (torch.cuda.is_available() and not args.disable_gpu) else "cpu"

    model = K2VAE(
        obs_dim=train_dataset.obs_dim,
        state_dim=args.state_dim,
        patch_size=args.patch_size,
        mlp_num_layers=args.mlp_num_layers,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_dropout=args.mlp_dropout,
        mlp_activation=args.mlp_activation,
        obs_context_len=args.obs_context_len,
        obs_forecast_len=args.obs_forecast_len,
        int_factor=args.int_factor,
        int_dropout=args.int_dropout,
        int_hidden_dim=args.int_hidden_dim,
        int_num_heads=args.int_num_heads,
        int_dff=args.int_dff,
        int_activation=args.int_activation,
        int_num_layers=args.int_num_layers,
        kf_init=args.kf_init,
        scaler_eps=args.scaler_eps,
        scaler_learnable=args.scaler_learnable,
    ).to(device)

    wandb.watch(model, log="all", log_freq=10)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for update in tqdm(range(args.num_updates)):

        # train
        model.train()
        x, y = train_dataset.sample(
            context_len=args.obs_context_len,
            forecast_len=args.obs_forecast_len,
            batch_size=args.batch_size,
            return_tensors="pt",
        )
        x, y = x.to(device), y.to(device)
        outputs = model(x=x, y=y)

        loss = (
            args.x_rec_weight * outputs.x_rec_loss +
            args.y_rec_weight * outputs.y_rec_loss +
            args.kl_beta * outputs.kl_loss
        )

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()

        wandb.log(
            {
            "train/x reconstruction loss": outputs.x_rec_loss.item(),
            "train/y prediction loss": outputs.y_rec_loss.item(),
            "train/kl loss": outputs.kl_loss.item(),
            "train/total loss": loss.item(),
            },
            step=update
        )

        if update % args.test_interval == 0:
            model.eval()
            # test
            with torch.no_grad():
                x, y = test_dataset.sample(
                context_len=args.obs_context_len,
                forecast_len=args.obs_forecast_len,
                batch_size=args.batch_size,
                return_tensors="pt",
                )
                x, y = x.to(device), y.to(device)
                outputs = model(x=x, y=y)

                loss = (
                    args.x_rec_weight * outputs.x_rec_loss +
                    outputs.y_rec_loss +
                    args.kl_beta * outputs.kl_loss
                )

                wandb.log(
                    {
                    "test/x reconstruction loss": outputs.x_rec_loss.item(),
                    "test/y prediction loss": outputs.y_rec_loss.item(),
                    "test/kl loss": outputs.kl_loss.item(),
                    "test/total loss": loss.item(),
                    },
                    step=update
                )

    save_dir = Path(args.log_dir) / args.run_id
    torch.save(model.state_dict(), save_dir / "model.pth")

    return model