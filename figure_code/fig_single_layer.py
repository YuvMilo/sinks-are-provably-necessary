"""Generate Figure 1 (softmax_relu_1H1D): single-layer softmax vs ReLU.

Trains a 1-head 1-layer softmax model and a 1-head 1-layer ReLU model,
then plots 4 panels: softmax mean, softmax std, ReLU mean, ReLU std.
"""

from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.data import TriggerMeanStream
from utils.models import SingleHeadAttention
from utils.training import train_model
from utils.plotting import plot_four_panel


def parse_args():
    p = argparse.ArgumentParser(description="Figure 1: single-layer experiments")
    p.add_argument("--d_model", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--trigger_pos", type=int, default=8, help="1-indexed trigger position for evaluation")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--test_batch_size", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=50000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="figures")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def attention_stats(model, d_model, seq_len, trigger_pos, test_batch_size, device):
    ds = TriggerMeanStream(d_model, seq_len, trigger_pos, sample_trigger=False)
    loader = DataLoader(ds, batch_size=test_batch_size, num_workers=0)
    x, _ = next(iter(loader))
    x = x.to(device)
    _, attn = model(x, return_attn=True)
    return attn.mean(dim=0).cpu().numpy(), attn.std(dim=0).cpu().numpy()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    print(f"Device: {device}")

    results = {}
    for attn_type in ["softmax", "relu"]:
        print(f"\nTraining {attn_type} model...")
        model = SingleHeadAttention(args.d_model, attn_type=attn_type).to(device)
        train_model(model, args.d_model, args.seq_len, args.trigger_pos,
                    residual=False, lr=args.lr, batch_size=args.batch_size,
                    max_steps=args.max_steps, seed=args.seed, device=device,
                    label=attn_type)
        print(f"Collecting attention stats ({attn_type})...")
        mean, std = attention_stats(model, args.d_model, args.seq_len,
                                    args.trigger_pos, args.test_batch_size, device)
        results[attn_type] = (mean, std)

    os.makedirs(args.output_dir, exist_ok=True)
    savepath = os.path.join(args.output_dir, "softmax_relu_1H1D.png")
    plot_four_panel(results["softmax"][0], results["softmax"][1],
                    results["relu"][0], results["relu"][1],
                    args.seq_len, savepath)
    print(f"\nSaved figure to {savepath}")


if __name__ == "__main__":
    main()
