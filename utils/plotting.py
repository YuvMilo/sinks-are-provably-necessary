"""Plotting utilities for attention heatmaps."""

from __future__ import annotations
from typing import List, Optional

import os
import numpy as np
import matplotlib.pyplot as plt


def _format_axes(ax, seq_len: int):
    ax.set_xlabel("Key positions", fontsize=22)
    ax.set_ylabel("Query positions", fontsize=22)
    ticks = list(range(1, seq_len, 2))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(i + 1) for i in ticks], fontsize=18)
    ax.set_yticklabels([str(i + 1) for i in ticks], fontsize=18)


def _save_fig(fig, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")


def plot_four_panel(softmax_mean: np.ndarray, softmax_std: np.ndarray,
                    relu_mean: np.ndarray, relu_std: np.ndarray,
                    seq_len: int, savepath: str):
    """4-panel figure: softmax mean/std and ReLU mean/std (Fig 1 in paper)."""
    fig, ax = plt.subplots(1, 4, figsize=(22, 5))
    titles = ["Softmax: mean", "Softmax: std", "ReLU: mean", "ReLU: std"]
    data = [softmax_mean, softmax_std, relu_mean, relu_std]
    labels = ["(a)", "(b)", "(c)", "(d)"]

    for i, (a, d, t) in enumerate(zip(ax, data, titles)):
        im = a.imshow(d, aspect="auto", interpolation="nearest",
                      vmin=0, vmax=1, cmap="Blues")
        a.set_title(t, fontsize=24)
        _format_axes(a, seq_len)
        a.text(0.5, -0.25, labels[i], transform=a.transAxes,
               ha="center", va="top", fontsize=22)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)
    _save_fig(fig, savepath)
    plt.close(fig)


def plot_all_heads_grid(attn_per_layer: List[np.ndarray],
                        num_layers: int, num_heads: int,
                        seq_len: int, savepath: str,
                        suptitle: Optional[str] = None):
    """Grid of attention heatmaps: one subplot per (layer, head).

    Works for any combination (1x4, 2x2, 4x4, etc.).
    """
    total = num_layers * num_heads
    if total <= 4:
        fig, axes = plt.subplots(1, total, figsize=(5.5 * total, 5))
        if total == 1:
            axes = [axes]
        else:
            axes = list(axes)
        flat_axes = axes
    else:
        fig, axes = plt.subplots(num_layers, num_heads,
                                 figsize=(5 * num_heads, 5 * num_layers))
        flat_axes = [axes[r][c] for r in range(num_layers)
                     for c in range(num_heads)]

    idx = 0
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            a = flat_axes[idx]
            attn = attn_per_layer[layer_idx][head_idx]
            im = a.imshow(attn, aspect="auto", interpolation="nearest",
                          vmin=0, vmax=1, cmap="Blues")
            a.set_title(f"Layer {layer_idx+1}, Head {head_idx+1}", fontsize=18)
            _format_axes(a, seq_len)
            fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
            idx += 1

    if suptitle:
        fig.suptitle(suptitle, fontsize=22, y=1.02)
    plt.tight_layout()
    _save_fig(fig, savepath)
    plt.close(fig)
