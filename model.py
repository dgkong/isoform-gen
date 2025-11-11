from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointerHead(nn.Module):
    """
    Cross-attend a single context vector to a suffix sequence.
    Inputs:
      x_context: [B, D]
      x_suffix : [B, S, D]
      mask     : [B, S]  (1=valid, 0=pad)
    Outputs:
      logits   : [B, S]  (unnormalized scores per suffix position)
      pred_idx : [B]     (argmax index in 0..S-1)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.ln_c = nn.LayerNorm(d_model)
        self.ln_s = nn.LayerNorm(d_model)
        self.wq   = nn.Linear(d_model, d_model, bias=False)
        self.wk   = nn.Linear(d_model, d_model, bias=False)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, x_context, x_suffix, mask = None, targets = None):
        # Normalize
        x_context = self.ln_c(x_context)    # [B, D]
        x_suffix = self.ln_s(x_suffix)      # [B, S, D]

        # Projections
        q = self.wq(x_context) * self.scale # [B, D]
        k = self.wk(x_suffix)               # [B, S, D]

        # Dot-product: (B,1,D) @ (B,D,S) -> (B,1,S) -> (B,S)
        logits = torch.matmul(q.unsqueeze(1), k.transpose(-2, -1)).squeeze(1) # [B, S]

        # Mask pads to -inf so softmax ignores them
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float("-inf"))

        pred_idx = logits.argmax(dim=-1)  # [B]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return logits, pred_idx, loss
