"""Generate multi-layer multi-head attention figures.

Trains softmax and ReLU multi-layer transformers and plots attention grids.
Produces figures used in the paper for 2-layer/2-head and 4-layer/4-head models.
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
from utils.models import MultiLayerTransformer
from utils.training import train_model
from utils.plotting import plot_all_heads_grid


def parse_args():
    p = argparse.ArgumentParser(description="Multi-layer multi-head attention figures")
    p.add_argument("--d_model", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--trigger_pos", type=int, default=8)
    p.add_argument("--num_heads", type=int, default=2)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_steps", type=int, default=50000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="figures")
    p.add_argument("--attn_types", nargs="+", default=["softmax", "relu"],
                   help="Attention types to train and plot")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def get_attention_single_example(model, d_model, seq_len, trigger_pos, device):
    ds = TriggerMeanStream(d_model, seq_len, trigger_pos,
                           sample_trigger=False, residual=True)
    loader = DataLoader(ds, batch_size=1, num_workers=0)
    x, _ = next(iter(loader))
    x = x.to(device)
    _, all_attn = model(x, return_attn=True)
    return [attn[0].cpu().numpy() for attn in all_attn]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    tag = f"{args.num_heads}H{args.num_layers}D"
    print(f"Device: {device}")
    print(f"Configuration: {args.num_layers} layers, {args.num_heads} heads")

    os.makedirs(args.output_dir, exist_ok=True)

    for attn_type in args.attn_types:
        print(f"\nTraining {attn_type} model...")
        model = MultiLayerTransformer(
            args.d_model, args.num_heads, args.num_layers,
            attn_type=attn_type
        ).to(device)
        train_model(model, args.d_model, args.seq_len, args.trigger_pos,
                    residual=True, lr=args.lr, batch_size=args.batch_size,
                    max_steps=args.max_steps, seed=args.seed, device=device,
                    label=attn_type)

        print(f"Collecting attention ({attn_type})...")
        attn_per_layer = get_attention_single_example(
            model, args.d_model, args.seq_len, args.trigger_pos, device)

        savepath = os.path.join(args.output_dir, f"{attn_type}_{tag}.png")
        plot_all_heads_grid(attn_per_layer, args.num_layers, args.num_heads,
                            args.seq_len, savepath)
        print(f"Saved {savepath}")


if __name__ == "__main__":
    main()
