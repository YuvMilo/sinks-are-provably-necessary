Code for reproducing the experiments in the paper "Attention Sinks Are Provably Necessary in Softmax Transformers: Evidence from Trigger-Conditional Tasks" [https://arxiv.org/abs/2603.11487].

## Setup

```bash
conda create -n sinks python=3.10 -y
conda activate sinks
pip install -r requirements.txt
```

## Reproducing the Figures

All commands should be run from the `repo/` directory.

### Single-Layer Softmax vs ReLU (Fig. 3 in the paper)

Trains a single-layer single-head softmax model and a ReLU model, then plots
mean and standard deviation of attention weights across 1000 test examples
(trigger at position 8).

```bash
python figure_code/fig_single_layer.py --output_dir figures
```

### Multi-Layer Multi-Head — 2 Layers, 2 Heads (Fig. 4 & Fig. 5)

Trains 2-layer 2-head softmax and ReLU models, then
plots per-head attention patterns on a single test input  with triger position 8.

```bash
# Crate for Softmax and Relu Figures
python figure_code/fig_multilayer.py --num_layers 2 --num_heads 2 --output_dir figures
```

### Multi-Layer Multi-Head — 4 Layers, 4 Heads (Fig. 6-8)

Trains 4-layer 4-head softmax and ReLU models, then
plots per-head attention patterns on a single test input with triger position 8.

```bash
# Crate for Softmax and Relu Figures
python figure_code/fig_multilayer.py --num_layers 4 --num_heads 4 --output_dir figures
```




