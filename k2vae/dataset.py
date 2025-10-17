import torch
import numpy as np
import pandas as pd
from pathlib import Path


class CustomDataset:

    def __init__(
        self,
        path: Path
    ):
        df = pd.read_csv(path)
        self.data = df.to_numpy(dtype=np.float32)

    def __len__(self):
        return self.data.shape[0]
    
    @property
    def obs_dim(self):
        return self.data.shape[1]

    def sample(
        self,
        context_len: int,
        forecast_len: int,
        batch_size: int,
        return_tensors: str="pt"
    ):
        chunk_length = context_len + forecast_len
        start_indexes = np.random.randint(0, len(self) - chunk_length + 1, size=batch_size)
        batch = np.stack([self.data[s: s+chunk_length] for s in start_indexes], axis=0)
        
        if return_tensors == "pt":
            batch = torch.as_tensor(batch)

        x = batch[:, :context_len, :]
        y = batch[:, context_len:, :]

        return x, y