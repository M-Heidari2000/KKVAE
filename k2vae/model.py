import torch
import torch.nn as nn
from collections import namedtuple
from torch.distributions import MultivariateNormal
from torch.distributions import kl_divergence
from einops import rearrange
from .components import (
    KPLayer,
    MLP,
    KalmanFilter,
    Integrator,
    Patcher,
    Scaler,
)


Outputs = namedtuple("Outputs", ["x_rec", "y_rec", "x_rec_loss", "y_rec_loss", "kl_loss"])


class K2VAE(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        patch_size: int,
        mlp_num_layers: int,
        mlp_hidden_dim: int,
        mlp_dropout: float,
        mlp_activation: str,
        obs_context_len: int,
        obs_forecast_len: int,
        int_factor: int,
        int_dropout: float,
        int_hidden_dim: int,
        int_num_heads: int,
        int_dff: int,
        int_activation: str,
        int_num_layers: int,
        kf_init: str,
        scaler_eps: float,
        scaler_learnable: bool,
    ):
        super().__init__()

        self.state_context_len = obs_context_len // patch_size
        self.state_forecast_len = obs_forecast_len // patch_size

        # scaler
        self.scaler = Scaler(
            obs_dim=obs_dim,
            eps=scaler_eps,
            learnable=scaler_learnable,
        )

        # patcher
        self.patcher = Patcher(
            patch_size=patch_size,
            state_dim=state_dim,
            obs_dim=obs_dim,
        )

        # koopman encoder
        self.encoder = MLP(
            input_dim=state_dim,
            output_dim=state_dim,
            hidden_dim=mlp_hidden_dim,
            num_hidden_layers=mlp_num_layers,
            dropout=mlp_dropout,
            activation=mlp_activation,
        )

        # koopman net
        self.koopman = KPLayer(
            state_dim=state_dim
        )

        # integrator
        self.integrator = Integrator(
            context_len=self.state_context_len,
            forecast_len=self.state_forecast_len,
            factor=int_factor,
            dropout=int_dropout,
            d_model=int_hidden_dim,
            n_heads=int_num_heads,
            d_ff=int_dff,
            activation=int_activation,
            state_dim=state_dim,
            num_layers=int_num_layers,
        )

        # kalman filter
        self.kalman_filter = KalmanFilter(
            state_dim=state_dim,
            init=kf_init,
        )

        # decoder
        self.decoder = MLP(
            input_dim=state_dim,
            output_dim=patch_size * obs_dim,
            hidden_dim=mlp_hidden_dim,
            num_hidden_layers=mlp_num_layers,
            dropout=mlp_dropout,
            activation=mlp_activation,
        )


    def forward(self, x, y):

        assert x.shape[1] % self.patcher.patch_size == 0
        assert y.shape[1] % self.patcher.patch_size == 0

        # normalize time series
        xn = self.scaler(x, mode="norm")
        # patching and linear embedding
        xpp = self.patcher(xn)
        # encoding
        xps = self.encoder(xpp)
        # koopman
        x_c, x_h = self.koopman(xps, forecast_len=self.state_forecast_len)
        # compute residuals
        x_res = xps - x_c
        u = self.integrator(x_res)
        # kalman filtering, why the initial state mean is xps instead of x_c ??
        z_dist = self.kalman_filter(x=xps[:, -1, :], u=u, y=x_h)
        z = z_dist.mean
        zp = z + u
        q_z = MultivariateNormal(loc=zp, covariance_matrix=z_dist.covariance_matrix)
        z_sample = q_z.rsample()
        
        x_rec = rearrange(self.decoder(x_c), "b n (p o) -> b (n p) o", p=self.patcher.patch_size)
        y_rec = rearrange(self.decoder(z_sample), "b n (p o) -> b (n p) o", p=self.patcher.patch_size)
        x_rec = self.scaler(x_rec, mode="denorm")
        y_rec = self.scaler(y_rec, mode="denorm")

        # x reconstruction loss
        x_rec_loss = ((x_rec - x)**2).sum(dim=-1).mean()
        # y reconstruction loss, P(y|z, x) log likelihood
        y_rec_loss = ((y_rec - y)**2).sum(dim=-1).mean()
        # kl loss
        p_z_mean = torch.zeros_like(q_z.loc)
        B, N, S = q_z.loc.shape
        p_z_cov = torch.eye(S, device=q_z.loc.device).expand(B, N, -1, -1)
        p_z = MultivariateNormal(loc=p_z_mean, covariance_matrix=p_z_cov)
        kl_loss = kl_divergence(q_z, p_z).mean()

        return Outputs(
            x_rec=x_rec,
            y_rec=y_rec,
            x_rec_loss=x_rec_loss,
            y_rec_loss=y_rec_loss,
            kl_loss=kl_loss,
        )