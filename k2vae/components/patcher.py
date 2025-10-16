import torch.nn as nn
from einops import rearrange


class Patcher(nn.Module):

    def __init__(
        self,
        patch_size: int,
        state_dim: int,
        obs_dim: int
    ):
        
        super().__init__()
        self.patch_size = patch_size
        self.state_dim = state_dim
        self.linear_projection = nn.Linear(obs_dim * patch_size, state_dim)


    def forward(self, x):
        # patch
        x = rearrange(x, "b (n s) o -> b n (s o)", s=self.patch_size)
        # linear projection
        x = self.linear_projection(x)
        return x