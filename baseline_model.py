import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions):
        # positions: (B, L) -> cos/sin: (B, L, D/2)
        freqs = torch.einsum("bs,d->bsd", positions.float(), self.inv_freq)
        return freqs.cos(), freqs.sin()


def apply_rotary_pos(x, cos, sin):
    # x: (..., D), cos, sin: (..., D/2)
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class BaselinePointerHead(nn.Module):
    def __init__(self, d_emb, d_model: int, num_roles: int = 4, role_dim: int = 64, dropout=0.1):
        super().__init__()
        self.ln_s = nn.LayerNorm(d_emb)
        self.hist_role_emb = nn.Embedding(num_roles, role_dim)
        self.tgt_role_emb = nn.Embedding(num_roles, role_dim)

        self.start_state = nn.Parameter(torch.randn(d_emb + role_dim) * 0.02)
        
        self.rope = RotaryPositionalEmbedding(d_model)

        self.wq = nn.Linear(d_emb + 2*role_dim, d_model, bias=False)
        self.wk = nn.Linear(d_emb, d_model, bias=False)
        self.scale = 1.0 / math.sqrt(d_model)

        self.unary_mlp = nn.Sequential(
            nn.Linear(d_emb, d_model), 
            nn.GELU(), 
            nn.Linear(d_model, num_roles)
        )

    def forward(self, x_ctx, x_suf, pos_ctx, pos_suf, role_ctx, role_tgt, mask_ctx=None, mask_suf=None, targets=None, rel_abundance=None):
        B, S, _ = x_suf.shape

        x_suf = x_suf.to(self.ln_s.weight.dtype)
        x_ctx = x_ctx.to(self.ln_s.weight.dtype)

        _, M, _ = x_ctx.shape
        
        summary = (
            self.start_state.to(dtype=x_ctx.dtype)
            .unsqueeze(0)
            .expand(B, -1)
            .clone()
        )

        if M > 0:
            if mask_ctx is None:
                mask_ctx = torch.ones(B, M, device=x_ctx.device, dtype=torch.long)
            else:
                mask_ctx = mask_ctx.to(dtype=torch.long)

            lengths = mask_ctx.sum(dim=1)    # (B,)
            has_hist = lengths > 0           # (B,)

            if has_hist.any():
                x_h = x_ctx[has_hist]        # (B_hist, M, d_emb)
                r_h = role_ctx[has_hist]     # (B_hist, M)
                len_h = lengths[has_hist]    # (B_hist,)

                last_idx_h = (len_h - 1).clamp(min=0)  # (B_hist,)
                bh = torch.arange(x_h.size(0), device=x_ctx.device)

                last_x = x_h[bh, last_idx_h]           # (B_hist, d_emb)
                last_role = r_h[bh, last_idx_h]        # (B_hist,)
                last_role_emb = self.hist_role_emb(last_role)  # (B_hist, role_dim)

                summary[has_hist] = torch.cat([last_x, last_role_emb], dim=-1)

        # Query input also includes the target role embedding
        q_in = torch.cat([summary, self.tgt_role_emb(role_tgt)], dim=-1)  # (B, d_emb + 2*role_dim)

        x_suf = self.ln_s(x_suf)

        anchor = pos_suf[:, 0].unsqueeze(1) - 1 # last context pos (B,1)  
        q_cos, q_sin = self.rope(anchor)
        k_cos, k_sin = self.rope(pos_suf)
        
        q = self.wq(q_in).unsqueeze(1) # (B,1,D)
        k = self.wk(x_suf) # (B, S, D)
        q = apply_rotary_pos(q, q_cos, q_sin)
        k = apply_rotary_pos(k, k_cos, k_sin)

        # Pairwise + Unary
        logits = torch.matmul(q, k.transpose(-2, -1)).squeeze(1) * self.scale # (B, S)
        
        unary_all = self.unary_mlp(x_suf) # (B, S, Roles)
        role_idx = role_tgt.view(B, 1, 1).expand(B, S, 1)
        unary = torch.gather(unary_all, dim=-1, index=role_idx).squeeze(-1)  # (B, S)
        
        logits = logits + unary

        if mask_suf is not None:
            logits = logits.masked_fill(mask_suf == 0, float("-inf"))

        loss = None
        if targets is not None:
            if rel_abundance is not None:
                loss = F.cross_entropy(logits, targets, reduction="none")
                loss = loss * rel_abundance
                loss = loss.sum() / rel_abundance.sum()
            else:
                loss = F.cross_entropy(logits, targets)
    
        return logits, logits.argmax(dim=-1), loss
