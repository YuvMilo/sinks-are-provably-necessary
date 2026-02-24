"""Data generation for the trigger-conditional averaging task."""

from __future__ import annotations

import torch
from torch.utils.data import IterableDataset


class TriggerMeanStream(IterableDataset):
    """Infinite stream of sequences for the trigger-conditional averaging task.

    Features per token x^(i) in R^n:
      - coord 0: BOS indicator (1 only for i==0)
      - coord 1: trigger indicator (1 only for i==j)
      - coords 2..n-2: i.i.d. Uniform(-1, 1) for ALL tokens including trigger
      - coord n-1: bias channel = 1 for content tokens, 0 for BOS and trigger

    Target y^(i):
      - For single-layer (residual=False): zero for all i != j; at i==j the
        elementwise mean of tokens at positions 1..j (inclusive).
      - For multi-layer with residual connections (residual=True): x^(i) for
        all i != j; at i==j, x^(j) + mean of tokens at positions 1..j (inclusive).

    If sample_trigger=True, j is sampled uniformly from {3..L} (1-indexed).
    Otherwise, j is fixed to trigger_pos (1-indexed).
    """

    def __init__(self, d_model: int, seq_len: int, trigger_pos: int,
                 sample_trigger: bool = False, residual: bool = False):
        super().__init__()
        self.n = d_model
        self.L = seq_len
        assert self.n >= 3 and self.L >= 4
        self.sample_trigger = sample_trigger
        self.residual = residual
        self.j_fixed = trigger_pos - 1  # convert to 0-indexed
        assert 1 < self.j_fixed < self.L

    def __iter__(self):
        while True:
            if self.sample_trigger:
                j = torch.randint(low=2, high=self.L, size=()).item()
            else:
                j = self.j_fixed

            x = torch.zeros(self.L, self.n, dtype=torch.float32)
            x[0, 0] = 1.0  # BOS

            if self.L > 1:
                if self.n > 3:
                    noise = torch.rand(self.L - 1, self.n - 3) * 2 - 1
                    x[1:, 2:-1] = noise
                x[1:, -1] = 1.0

                # trigger token: set trigger indicator, keep i.i.d. noise, clear BOS and bias
                x[j, 0] = 0.0
                x[j, 1] = 1.0
                x[j, -1] = 0.0

            if self.residual:
                y = x.clone()
                if j >= 1:
                    y[j] = x[j] + x[1:j+1].mean(dim=0)
            else:
                y = torch.zeros_like(x)
                if j >= 1:
                    y[j] = x[1:j+1].mean(dim=0)

            yield x, y
