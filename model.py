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


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, rope_cos, rope_sin, mask=None):
        B, L, D = x.shape
        if L == 0: 
            return x

        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        # (B, H, L, D_head)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        # (B, L, D/2) -> (B, H, L, D_head/2)
        cos = rope_cos.view(B, L, self.num_heads, self.head_dim//2).transpose(1, 2)
        sin = rope_sin.view(B, L, self.num_heads, self.head_dim//2).transpose(1, 2)
        q = apply_rotary_pos(q, cos, sin)
        k = apply_rotary_pos(k, cos, sin)
        
        # (B, H, L, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask.view(B, 1, 1, L) == 0, float("-inf"))
            
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        # (B, H, L, D_head)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o_proj(out)
        return self.proj_drop(out)


class HistoryEncoder(nn.Module):
    def __init__(self, input_dim, d_model, rope_module, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.rope = rope_module
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": SelfAttention(d_model),
                "ln1": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, 4 * d_model), 
                    nn.GELU(), 
                    nn.Linear(4 * d_model, d_model)
                ),
                "ln2": nn.LayerNorm(d_model),
                "drop_ffn": nn.Dropout(dropout),
            }) for _ in range(num_layers)
        ])

        self.start_state = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, x, positions, mask=None):
        B, M, _ = x.shape
        if M == 0:
            return self.start_state.to(dtype=x.dtype).expand(B, -1)

        x = self.input_proj(x)
        if mask is None:
            mask = torch.ones(B, M, device=x.device)
        
        lengths = mask.sum(dim=1)              # (B,)
        has_hist = lengths > 0                 # (B,)

        summary = x.new_empty(B, x.size(-1))   # (B,D)

        # Process only rows that actually have history
        x_h      = x[has_hist]               # (B_hist,M,D)
        pos_h    = positions[has_hist]       # (B_hist,M)
        mask_h   = mask[has_hist]            # (B_hist,M)

        cos_h, sin_h = self.rope(pos_h)      # (B_hist,M,D/2)

        for layer in self.layers:
            x_h = x_h + layer["attn"](layer["ln1"](x_h), cos_h, sin_h, mask=mask_h)
            x_h = x_h + layer["drop_ffn"](layer["ffn"](layer["ln2"](x_h)))

        last_idx_h = (mask_h.sum(dim=1).long() - 1).clamp(min=0)
        batch_idx_h = torch.arange(x_h.size(0), device=x.device)
        summary[has_hist] = x_h[batch_idx_h, last_idx_h]

        # Rows with no history get start_state
        summary[~has_hist] = self.start_state.to(dtype=x.dtype)

        return summary


class PointerHead(nn.Module):
    def __init__(self, d_emb, d_model: int, num_roles: int = 4, role_dim: int = 64, dropout=0.1):
        super().__init__()
        self.ln_s = nn.LayerNorm(d_emb)
        self.hist_role_emb = nn.Embedding(num_roles, role_dim)
        self.tgt_role_emb = nn.Embedding(num_roles, role_dim)
        
        self.rope = RotaryPositionalEmbedding(d_model)
        self.encoder = HistoryEncoder(d_emb + role_dim, d_model, self.rope, dropout=dropout)
        
        self.wq = nn.Linear(d_model + role_dim, d_model, bias=False)
        self.wk = nn.Linear(d_emb, d_model, bias=False)
        self.scale = 1.0 / math.sqrt(d_model)

        self.unary_mlp = nn.Sequential(
            nn.Linear(d_emb, d_model), 
            nn.GELU(), 
            nn.Linear(d_model, num_roles)
        )

    def forward(self, x_ctx, x_suf, pos_ctx, pos_suf, role_ctx, role_tgt, mask_ctx=None, mask_suf=None, targets=None, rel_abundance=None):
        B, S, _ = x_suf.shape

        # Encode History
        h_emb = self.hist_role_emb(role_ctx)
        summary = self.encoder(torch.cat([x_ctx, h_emb], dim=-1), pos_ctx, mask_ctx)
        q_in = torch.cat([summary, self.tgt_role_emb(role_tgt)], dim=-1)

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
