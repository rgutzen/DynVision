# Benchmarking & Computational Performance

This reference documents the computational cost of different modeling choices
and provides the default training configuration tables from the published
DynVision manuscript.

## Default Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Temporal Dynamics** | | |
| Time steps | 20 | Simulation timesteps |
| Resolution ($dt$) | 2 ms | Integration time step |
| Time constant ($\tau$) | 5 ms | Neural time constant |
| $\Delta_{FF}$ | 0 ms | Feedforward delay (engineering time) |
| $\Delta_{SK}$ | 0 ms | Skip‑connection delay (engineering time) |
| $\Delta_{RC}$ | 6 ms | Recurrent connection delay |
| Integration strategy | Additive | Activity integration |
| Dynamics solver | Euler | Numerical ODE solver |
| Recurrence target | Output | Recurrence applied to layer output |
| **Architecture** | | |
| Skip connections | V1→V4, V2→IT | |
| Feedback | Disabled | V1←V4, V2←IT |
| Normalization | None | No BatchNorm / LayerNorm |
| **Training** | | |
| Epochs | 200 | |
| Dataset | ImageNette | 224 px, 10 classes |
| Batch size | 192 | |
| Learning rate | 0.0008 | |
| LR schedule | CosineAnnealingLR | $T_{\max}=500$ |
| Recurrent LR factor | 0.2 | 5× smaller LR for recurrent |
| Feedback LR factor | 0.5 | 2× smaller LR for feedback |
| Optimizer | Adam | |
| Weight decay | 0.0005 | |
| **Loss** | | |
| Primary | Cross‑entropy | |
| Auxiliary | Activity Loss (×0.0002) | |
| Loss reaction time | 4 ms | |
| **Optimization** | | |
| Gradient accumulation | 4 batches | Effective batch 768 |
| Precision | bf16‑mixed | |
| Gradient clipping | Norm, 1.0 | |
| **Hardware** | | |
| GPU | NVIDIA A100 | |
| CPU | 16 cores | |

## Computational Resource Demands

Values show the **change relative to the default** DyRCNNx8 configuration
(full recurrence, output target, 20 timesteps, no feedback). Reported on
NVIDIA A100‑SXM4‑80 GB, batch size 192, averaged over 3 seeds.

| Parameter | Value | Δ Time / Epoch (min) | Δ GPU Mem. (GB) | Δ # Params |
|-----------|-------|----------------------|-----------------|------------|
| *Default* | | 1.00 | 48 | 8,534,794 |
| **Recurrence Type** | | | | |
| | none | −0.2 | −6 | −3,398,410 |
| | self | −0.1 | +1 | −3,398,402 |
| | depthpointwise | +0.2 | +6 | −3,016,254 |
| | pointdepthwise | +0.2 | +6 | −3,016,254 |
| | local | +0.6 | +11 | −3,398,338 |
| | localdepthwise | +0.9 | +23 | −3,387,384 |
| **Recurrence Target** | | | | |
| | input | 0 | −4 | −1,759,607 |
| | middle | 0 | +6 | 0 |
| **Skip** | False | −0.2 | −4 | −93,345 |
| **Feedback** | Additive | +0.1 | +6 | +92,768 |
| | Multiplicative | +0.1 | +14 | +92,768 |
| **Timesteps** | 8 | −0.6 | −25 | 0 |
| | 26 | +0.3 | +13 | 0 |
| **Unrolling** | biological | +0.2 | +43 | 0 |

## Key Takeaways

- **52 % speedup over CORnet‑RT**: reimplementation runs at 8.86 s/epoch vs.
  original 13.51 s/epoch on A100 (averaged over first 80 epochs).
- Engineering vs. biological time: engineering time reduces epoch time by
  ~29 % and GPU memory from 2.39 GB to 2.13 GB in a representative DyRCNNx8
  training run.
- The Dense recurrence targeting the *middle* increases GPU memory by ~6 GB
  relative to *output* despite identical channel dimensions — the autograd
  engine must retain both first‑convolution output and recurrence output for
  the second convolution's backward pass.
- Disabling the activity loss frees ~19 GB GPU memory (no per‑timestep
  activation norms to track).
- FFCV loaders can provide 1.2–7× epoch‑speed improvements, but benefits
  depend on dataset size and compute‑ vs. I/O‑bound workload.

## See Also

- [Default Training Configuration](../explanation/engineering-vs-biological-time.md) — eng. vs. bio. time conversion
- [FFCV Dependency Guide](../development/dependencies/ffcv.md)
- [Model Testing (How‑to)](../user-guide/model-testing.md)
