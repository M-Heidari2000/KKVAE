import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


class KalmanFilter(nn.Module):
    def __init__(self, state_dim, init='identity'):

        super().__init__()
        self.state_dim = state_dim

        # Initialize Kalman filter parameters
        if init == 'identity':
            self.A = nn.Parameter(torch.eye(state_dim, state_dim))
            self.B = nn.Parameter(torch.eye(state_dim, state_dim))
        else:
            self.A = nn.Parameter(torch.randn(state_dim, state_dim))
            self.B = nn.Parameter(torch.randn(state_dim, state_dim))

        self.C = nn.Parameter(torch.eye(state_dim, state_dim))

        # Learnable covariance matrices Ns, No
        self.LNs = nn.Parameter(torch.tril(torch.eye(state_dim, state_dim)))
        self.LNo = nn.Parameter(torch.tril(torch.eye(state_dim, state_dim)))

    @property
    def Ns(self):
        return self.LNs @ self.LNs.T
    
    @property
    def No(self):
        return self.LNo @ self.LNo.T


    def dynamics_update(self, x_prev, P_prev, u):
        x_pred = torch.einsum("ij,bj->bi", self.A, x_prev) + torch.einsum("ij,bj->bi", self.B, u)
        P_pred = torch.einsum("ij,bjk,kl->bil", self.A, P_prev, self.A.transpose(-1, -2)) + self.Ns
        return x_pred, P_pred

    def measurement_update(self, x_pred, P_pred, y):
        S = torch.einsum("ij,bjk,kl->bil", self.C, P_pred, self.C.transpose(-1, -2)) + self.No
        G = torch.einsum("bij,jk,bkl->bil", P_pred, self.C.transpose(-1, -2), torch.linalg.pinv(S))
        innovation = y - torch.einsum("ij,bj->bi", self.C, x_pred)
        x_update = x_pred + torch.einsum("bij,bj->bi", G, innovation)
        P_update = P_pred - torch.einsum("bij,jk,bkl->bil", G, self.C, P_pred)
        return x_update, P_update


    def one_step(self, x_prev, P_prev, u, y):
        x_pred, P_pred = self.dynamics_update(x_prev=x_prev, P_prev=P_prev, u=u)
        x_update, P_update = self.measurement_update(x_pred=x_pred, P_pred=P_pred, y=y)
        return x_update, P_update


    def keep_positive_definite(self, tensor):
        tensor = tensor + tensor.transpose(-1, -2)
        eigvals, eigvecs = torch.linalg.eigh(tensor)
        eigvals_clamped = torch.clamp(eigvals, min=1e-6)  # [B, pred_len, state_dim]
        fixed_tensor = eigvecs @ torch.diag_embed(eigvals_clamped) @ eigvecs.transpose(-1, -2)
        fixed_tensor = fixed_tensor + fixed_tensor.transpose(-1, -2)
        return fixed_tensor


    def forward(self, x, u, y):
        # x is the initial state mean
        batch_size, seq_len, _ = u.shape
        # Initial covariance matrix
        P_t = torch.eye(self.state_dim, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)

        prediction = []
        covariance = []

        # Iterate over the sequence
        for t in range(seq_len):
            x, P_t = self.one_step(x_prev=x, u=u[:, t, :], y=y[:, t, :], P_prev=P_t)
            P_t = 0.5 * (P_t + P_t.transpose(-2, -1))
            prediction.append(x)
            covariance.append(P_t)

        # Stack predictions and covariances
        prediction = torch.stack(prediction, dim=1)
        covariance = torch.stack(covariance, dim=1)

        # Create a multivariate normal distribution for the predictions
        try:
            dist = MultivariateNormal(loc=prediction.to(x.device), covariance_matrix=covariance)
        except:
            covariance = self.keep_positive_definite(covariance)
            dist = MultivariateNormal(loc=prediction.to(x.device), covariance_matrix=covariance)

        return dist

