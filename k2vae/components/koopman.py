import torch
import torch.nn as nn
from typing import Optional


class MLP(nn.Module):

    def __init__(self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int]=128,
        num_hidden_layers: Optional[int]=2,
        dropout: Optional[float]=0.05,
        activation: Optional[str]='tanh'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError

        layers = [
            nn.Linear(self.input_dim, self.hidden_dim),
            self.activation,
            nn.Dropout(self.dropout)
        ]

        for _ in range(self.num_hidden_layers - 1):
            layers += [
                nn.Linear(self.hidden_dim, self.hidden_dim),
                self.activation,
                nn.Dropout(dropout)
            ]

        layers += [nn.Linear(hidden_dim, output_dim)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class KPLayer(nn.Module):

    def __init__(self, state_dim: int):
        super().__init__()
        K_init = torch.randn(state_dim, state_dim)
        U, _, Vh = torch.linalg.svd(K_init)  # stable initialization
        self.K_global = nn.Parameter(U @ Vh)

    def forward(self, z, forecast_len: int):
        # B: batch size, n: context length, E: state dim, m: forecast len
        B, n, E = z.shape
        m = forecast_len
        z_back, z_fore = z[:, :-1, :], z[:, 1:, :]
        # solve Z_back @ K_loc = Z_fore with least-squares
        K_local = torch.linalg.lstsq(z_back, z_fore).solution
        # K = K_local + K_global
        K = K_local + self.K_global.unsqueeze(0).expand(B, -1, -1)

        if torch.isnan(K).any():
            print('Encounter K with nan, replace K by identity matrix')
            K = torch.eye(K.shape[1], device=K.device).expand(B, -1, -1)

        x_rec = torch.empty_like(z, device=z.device)
        x_forecast = torch.empty((B, m, E), device=z.device)
        x_rec[:, 0, :] = z[:, 0, :]
        dummy = z[:, 0, :]

        for i in range(1, n):
            dummy = torch.einsum("bi,bij->bj", dummy, K)
            x_rec[:, i, :] = dummy
        
        for i in range(0, m):
            dummy = torch.einsum("bi,bij->bj", dummy, K)
            x_forecast[:, i, :] = dummy

        return x_rec, x_forecast