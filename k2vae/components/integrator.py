import torch
from einops import rearrange
from torch import nn
from transformer.attention import FullAttention, AttentionLayer
from transformer.encdec import Encoder, EncoderLayer


class Integrator(nn.Module):
    def __init__(
        self,
        context_len: int,
        forecast_len: int,
        factor: int,
        dropout: float,
        d_model: int,
        n_heads: int,
        d_ff: int,
        activation: str,
        state_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.context_len = context_len
        self.forecast_len = forecast_len

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        # Adaption Head
        self.adaptation = nn.Linear(state_dim, d_model)

        # Prediction Head
        self.head = nn.Linear(context_len * d_model, forecast_len * state_dim)

    def forward(self, x):
        # Encoder
        B, n, state_dim = x.shape        
        x_enc = self.adaptation(x)
        enc_out, attns = self.encoder(x_enc)

        # Decoder
        enc_out = rearrange(enc_out, "b l c -> b (l c)")
        dec_out = self.head(enc_out)
        dec_out = rearrange(dec_out, "b (p c) -> b p c", c=state_dim)

        return dec_out