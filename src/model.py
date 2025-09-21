import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import Config


class SimpleTransformer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        # Embedding
        self.embed = nn.Parameter(torch.randn(config.d_model, config.p + 1) / np.sqrt(config.d_model))
        self.pos_embed = nn.Parameter(torch.randn(3, config.d_model) / np.sqrt(config.d_model))

        # Attention
        d_head = config.d_model // config.num_heads
        self.W_Q = nn.Parameter(torch.randn(config.num_heads, d_head, config.d_model) / np.sqrt(config.d_model))
        self.W_K = nn.Parameter(torch.randn(config.num_heads, d_head, config.d_model) / np.sqrt(config.d_model))
        self.W_V = nn.Parameter(torch.randn(config.num_heads, d_head, config.d_model) / np.sqrt(config.d_model))
        self.W_O = nn.Parameter(torch.randn(config.d_model, config.d_model) / np.sqrt(config.d_model))

        # MLP
        self.W_in = nn.Parameter(torch.randn(config.d_mlp, config.d_model) / np.sqrt(config.d_model))
        self.W_out = nn.Parameter(torch.randn(config.d_model, config.d_mlp) / np.sqrt(config.d_model))
        self.b_in = nn.Parameter(torch.zeros(config.d_mlp))
        self.b_out = nn.Parameter(torch.zeros(config.d_model))

        # Unembed
        self.W_U = nn.Parameter(torch.randn(config.d_model, config.p + 1) / np.sqrt(config.p + 1))

        self.register_buffer('mask', torch.tril(torch.ones(3, 3)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embed + pos
        x = torch.einsum('dbp -> bpd', self.embed[:, x]) + self.pos_embed

        # Attention
        q = torch.einsum('ihd,bpd->biph', self.W_Q, x)
        k = torch.einsum('ihd,bpd->biph', self.W_K, x)
        v = torch.einsum('ihd,bpd->biph', self.W_V, x)

        scores = torch.einsum('biph,biqh->biqp', k, q) / np.sqrt(self.config.d_model // self.config.num_heads)
        scores = scores.masked_fill(self.mask == 0, -1e10)
        attn = F.softmax(scores, dim=-1)

        z = torch.einsum('biph,biqp->biqh', v, attn)
        z = z.reshape(x.shape[0], x.shape[1], -1)
        attn_out = torch.einsum('df,bqf->bqd', self.W_O, z)

        # Residual
        x = x + attn_out

        # MLP
        mlp_out = F.relu(torch.einsum('md,bpd->bpm', self.W_in, x) + self.b_in)
        mlp_out = torch.einsum('dm,bpm->bpd', self.W_out, mlp_out) + self.b_out

        # Residual
        x = x + mlp_out

        # Unembed
        return x @ self.W_U
