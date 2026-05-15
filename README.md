# OpenGNS

**An Open Dataset of the Gradient Noise Scale across Vision, Language, and Diffusion Models**


## Overview

The **Gradient Noise Scale (GNS)** measures the signal-to-noise ratio of gradients in stochastic gradient descent. As shown by McCandlish et al. (2018), it can approximate the **Critical Batch Size (CBS)** — the batch size at which scaling efficiency drops below a threshold — which is directly relevant to efficient distributed training of large models.

Despite the GNS appearing in technical reports of frontier models (e.g., GPT-3), no open dataset of GNS measurements across modern architectures existed. **OpenGNS** closes this gap.

We release over **400 training trajectories** spanning three domains, eleven model sizes, and dense batch-size/learning-rate grids, all trained under **Maximal Update Parameterization (muP)**. Alongside the dataset, we present an empirical study of how the GNS depends on hyperparameters, model scale, and training progress, and how it relates to the CBS.

---

## Dataset

The dataset is hosted on [Hugging Face](https://huggingface.co/datasets/olympique-marcel/OpenGNS) and split into one Parquet file per workload. In total there are **433 training runs**: 180 ResNet configurations, 178 GPT configurations, and 75 DiT configurations.

### Experimental Setup

| Hyperparameter | ResNet18 | GPT | DiT |
|---|---|---|---|
| Dataset | CIFAR-10 | FineWeb | ImageNet |
| Training data | 10M images | 20B tokens | 51M images |
| Validation data | 10,000 images | 100M tokens | 40,504 images |
| Model widths | [1×, 2×, 4×] | [256, 512, 1024, 2048, 2560] | [144, 288, 576] |
| Parameters | 11M, 45M, 179M | 22M, 64M, 203M, 707M, 1B | 11M, 43M, 169M |
| Batch sizes | 2⁵–2¹⁴ images | 2¹⁵–2²¹ tokens | 2⁶–2¹⁰ images |
| Learning rates | 10⁻⁵–10⁻¹ | 2⁻¹¹–2⁻⁶ | 2⁻¹³–2⁻⁹ |
| Optimizer | AdamW | AdamW | AdamW |
| LR schedule | Cosine (epoch) | Cosine (step) | Constant |

## Quick Start

```python
import pandas as pd
import matplotlib.pyplot as plt

nlp = pd.read_parquet("nlp.parquet")

SEQUENCE_LENGTH = 1024
run = nlp[(nlp["width"] == 1024) & (nlp["batch_size"] == 1024) & (nlp["peak_lr"] == 2**-7)]

plt.plot(run["samples_seen"] * SEQUENCE_LENGTH, run["gns"])
plt.xlabel("Tokens seen")
plt.ylabel("GNS")
plt.show()
```

![GNS over training (GPT/FineWeb)](assets/nlp.png)
---

### Files and Columns

**`cv.parquet`** — 11.2M rows, ResNet18 on CIFAR-10

| Column | Description |
|---|---|
| `iteration` | Training step |
| `gns` | Simplified gradient noise scale B_simple |
| `gns_norm` | Squared gradient norm \|G\|² |
| `gns_var` | Trace of per-example gradient covariance tr(Σ) |
| `train_loss` / `val_loss` / `test_loss` | Losses |
| `train_acc` / `val_acc` / `test_acc` | Accuracies |
| `width` | Model width multiplier (1, 2, 4) |
| `batch_size` | Global batch size (images) |
| `lr` | Learning rate |
| `samples_seen` | Total training samples processed |

**`nlp.parquet`** — 27.5M rows, GPT on FineWeb

| Column | Description |
|---|---|
| `iteration` | Training step |
| `gns` / `gns_norm` / `gns_var` | GNS and its components |
| `train/loss` / `val/loss` | Losses |
| `width` | Model width (256, 512, 1024, 2048, 2560) |
| `batch_size` | Global batch size |
| `peak_lr` | Peak learning rate hyperparameter |
| `lr` | Instantaneous learning rate (varies with cosine schedule) |
| `lr_schedule` | LR schedule type (`cosine`) |
| `seed` | Random seed |
| `samples_seen` | Total training samples processed (multiply by sequence length 1024 to get in tokens) |

**`diffusion.parquet`** — 265K rows, DiT on ImageNet

| Column | Description |
|---|---|
| `iteration` | Training step |
| `gns` / `gns_norm` / `gns_var` | GNS and its components |
| `train/loss` / `val/loss` | Losses on ImageNet |
| `coco_val/loss` | Validation loss on COCO |
| `heads` | Number of attention heads (model size identifier) |
| `log_lr` | Learning rate on log₂ scale |
| `batch_size` | Global batch size (images) |
| `samples_seen` | Total training images processed |

### GNS Columns Explained

Each row records three quantities that together define the GNS:

```
GNS (B_simple) = gns_var / gns_norm = tr(Σ) / |G|²
```

`gns` is the smoothed GNS estimate. `gns_norm` and `gns_var` are the raw components, released separately so users can reconstruct or study them independently.

---

### Training Code

The training code for each workload is based on the following open-source repositories, all adapted to use Maximal Update Parameterization (muP) for hyperparameter transfer across model scales.

- **GPT (muP):** [EleutherAI/nanoGPT-mup](https://github.com/EleutherAI/nanoGPT-mup)  
  A muP-adapted fork of nanoGPT used for the language modeling experiments on FineWeb. GNS measurements were added to the training loop on top of this codebase.

- **DiT (muP):** [ML-GSAI/Scaling-Diffusion-Transformers-muP](https://github.com/ML-GSAI/Scaling-Diffusion-Transformers-muP)  
  A muP-adapted Diffusion Transformer (DiT) implementation used for the image generation experiments on ImageNet. GNS tracking was integrated into this training setup.

- **ResNet (muP):** [microsoft/mup — ResNet example](https://github.com/microsoft/mup/tree/main/examples/ResNet)  
  The official muP ResNet example from Microsoft, used as the basis for the vision classification experiments on CIFAR-10 with ResNet18.


---

## Using `gns_utils.py` in Your Own Training Loop

`gns_utils.py` provides two classes that together measure the **Gradient Noise Scale** during DDP training.

| Class | Role |
|---|---|
| `GradientNoiseScaleHook` | Hooks into DDP's gradient communication to capture per-GPU and global gradient squared norms each step |
| `GradientNoiseScale` | Accumulates those norms into an EMA and computes `GNS = tr(Σ) / ‖G‖²` |

### Minimal Example

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import gns_utils

# 1. Wrap your model in DDP first — the hook requires a DDP module
dist.init_process_group("nccl")
model = MyModel().to(device)
model = DDP(model, device_ids=[device])

# 2. Attach the communication hook (must happen after DDP wrapping)
gns_hook = gns_utils.GradientNoiseScaleHook(model)

# 3. Create the EMA tracker (beta controls smoothing; 0.995 is a good default)
gns_stats = gns_utils.GradientNoiseScale(beta=0.995)

# Training loop
for x, y in dataloader:
    loss = compute_loss(model, x, y)
    optimizer.zero_grad()
    loss.backward()

    # 4. After backward, before optimizer.step — collect gradient norms
    #    get_stats() returns (sq_norm_small_batch, sq_norm_large_batch)
    #    and resets the hook's internal buffers for the next step
    sq_norm_small, sq_norm_large = gns_hook.get_stats()

    n_small = x.shape[0]                          # per-GPU batch size
    n_large = x.shape[0] * dist.get_world_size()  # global batch size

    gns_stats.update(sq_norm_small, sq_norm_large, n_small, n_large)

    optimizer.step()

    # 5. Read out the current GNS and its components
    gns = gns_stats.get_gns()                        # B_simple = tr(Σ) / ‖G‖²
    grad_norm, grad_var = gns_stats.get_stats()      # debiased ‖G‖² and tr(Σ)
    print(f"GNS: {gns:.3f}  |G|²: {grad_norm:.3f}  tr(Σ): {grad_var:.3f}")
```

### Notes

- `GradientNoiseScaleHook` **requires a DDP-wrapped model**; it will raise `ValueError` on plain `nn.Module`.
- `get_stats()` on the hook internally calls `all_reduce` across ranks and resets buffers — call it exactly once per step, after `loss.backward()`.
---

## Repository Structure

```
OpenGNS/
├── gns-utils.py                    # Shared utilities (GNS estimator) 
└── plots/
    ├── cbs_gns_plots/              # GNS vs. Critical Batch Size plots
    │   ├── plot_CBS_GNS_joint_cv.ipynb
    │   ├── plot_CBS_GNS_joint_nlp.ipynb
    │   └── plot_CBS_GNS_joint_diffusion.ipynb
    ├── gns_plots/                  # GNS component analysis (NLP)
    │   ├── nlp_gns_dependency.ipynb
    │   └── nlp_gns_norm_var.ipynb
    └── temperature_comparison/     # GNS as temperature of training
        ├── cv_temperature_comparison.ipynb
        ├── nlp_temperature_comparison.ipynb
        └── diffusion_temperature_comparison.ipynb
```

---

## References

McCandlish et al. (2018). *An Empirical Model of Large-Batch Training.*  
Yang et al. (2021). *Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (muP).* NeurIPS.  
Zhang et al. (2024). *How Does Critical Batch Size Scale in Pre-Training?* OPT Workshop.
