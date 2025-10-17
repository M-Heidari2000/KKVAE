import torch
import torch.nn as nn


class Scaler(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        eps: float=1e-5,
        learnable: bool=True,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.eps = eps
        self.learnable = learnable

        if self.learnable:
            self._init_params()

    def _init_params(self):
        self.weight = nn.Parameter(torch.ones(self.obs_dim))
        self.bias = nn.Parameter(torch.zeros(self.obs_dim))

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.learnable:
            x = x * self.weight
            x = x + self.bias
        return x

    def _denormalize(self, x):
        if self.learnable:
            x = x - self.bias
            x = x / (self.weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x