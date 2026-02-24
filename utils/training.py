"""Training loop for attention models."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import TriggerMeanStream


def train_model(model: nn.Module, d_model: int, seq_len: int,
                trigger_pos: int, residual: bool = False,
                lr: float = 1e-3, batch_size: int = 128,
                max_steps: int = 10000, stop_loss: float = 5e-3,
                seed: int = 0, device: str = "cpu",
                label: str = "") -> nn.Module:
    """Train a model on the trigger-conditional averaging task.

    Args:
        model: The model to train (already on device).
        d_model: Token dimension.
        seq_len: Sequence length.
        trigger_pos: 1-indexed trigger position (used only for fixed-trigger eval).
        residual: Whether the model uses residual connections (changes targets).
        lr: Learning rate.
        batch_size: Training batch size.
        max_steps: Maximum number of gradient steps.
        stop_loss: Stop when L-infinity error on a batch falls below this.
        seed: Random seed.
        device: Torch device string.
        label: Label for the progress bar.

    Returns:
        The trained model.
    """
    torch.manual_seed(seed)
    ds = TriggerMeanStream(d_model, seq_len, trigger_pos,
                           sample_trigger=True, residual=residual)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0,
                        pin_memory=(device != "cpu"))

    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95))

    pbar = tqdm(range(max_steps), desc=f"Training ({label})", unit="step")
    it = iter(loader)
    for step in pbar:
        x, y = next(it)
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = criterion(pred, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        l_infty = torch.max(torch.abs(pred - y)).item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", l_infty=f"{l_infty:.4f}")

        if l_infty < stop_loss:
            break

    return model
