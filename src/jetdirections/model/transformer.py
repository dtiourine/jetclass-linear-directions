import torch
import torch.nn as nn
from jetdirections.config import INPUT_DIM, MODEL_DIM, NUM_HEADS, NUM_LAYERS, NUM_CLASSES, DROPOUT
from einops import rearrange, repeat


class Block(nn.Module):
    def __init__(
        self,
        d_model: int = MODEL_DIM,
        num_heads: int = NUM_HEADS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=False
        )

        self.mlp = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=d_model * 4, out_features=d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        attn_out, attn_weights = self.attn(
            self.ln1(x),
            self.ln1(x),
            self.ln1(x),
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        x = x + attn_out

        mlp_out = self.mlp(self.ln2(x))
        x = x + mlp_out

        return x, attn_weights


class JetTagTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        embed_dim: int = MODEL_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList(
            [
                Block(d_model=embed_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, mask=None, return_attention=False):
        """
        Args:
            x: (batch, input_dim, seq_len) - particle features
            mask: (batch, seq_len) - True for padding positions
            return_attention: bool - whether to return attention weights for interp

        Returns:
            logits: (batch, num_classes)
            attention_weights: list of (batch, num_heads, seq, seq) if return_attention=True
        """
        batch_size, _, seq_len = x.size()

        x = rearrange(x, "batch input_dim seq -> seq batch input_dim")

        x = self.embed(x)

        cls_tokens = repeat(
            self.cls_token, "1 1 embed_dim -> 1 batch embed_dim", batch=batch_size
        )
        x = torch.cat([cls_tokens, x], dim=0)

        if mask is not None:
            cls_mask = torch.zeros(batch_size, 1, dtype=mask.dtype, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)

        attention_weights = []
        for block in self.blocks:
            x, attn_w = block(x, key_padding_mask=mask)
            if return_attention:
                attention_weights.append(attn_w)

        x = x[0]  # Extract CLS token representation

        x = self.final_norm(x)
        logits = self.classifier(x)

        if return_attention:
            return logits, attention_weights
        return logits