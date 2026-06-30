# Evaluation Metrics

This reference describes the evaluation metrics DynVision computes during
training and testing, and how they generalise to temporally extended inputs.

## Description

DynVision's metrics are automatically computed and aggregated across models
and testing experiments within the Snakemake workflow. All metrics respect
the temporal dimension of model outputs.

## Accuracy

Classification accuracy at each timestep:

$$\text{Accuracy}(t) = \frac{1}{N}\sum_{n=1}^{N} \left[\mathrm{argmax}_i\, c_i(t) = y_n\right]$$

- **During training** — computed only for timesteps where the input has
  propagated through the network to reach the classifier (excluding residual
  timesteps and the `loss_reaction_time` offset).
- **During testing** — computed for *all* timesteps, enabling analysis of
  classification dynamics during both stimulus presentation and subsequent
  null‑input periods.
- **Top‑$k$ accuracy** follows analogously.

## Confidence

Softmax‑based certainty measures:

- **Target confidence** — softmax probability assigned to the *correct* class:
  $\mathrm{softmax}(c)_{y_n}(t)$.
- **Guess confidence** — maximum softmax probability:
  $\max_i \mathrm{softmax}(c)_i(t)$.

## Average Response

Recorded via a `record` operation placed in the `layer_operations` list.
By default activations are captured after each layer's nonlinearity.
Spatially‑averaged per‑layer responses provide a first‑order proxy for
comparison with electrophysiological data such as ECoG.

## See Also

- [Layer Operations](layer-operations.md) — where `record` fits
- [Benchmarking](benchmarking.md) — computational cost of metric tracking
- [Model Testing (How‑to)](../user-guide/model-testing.md)
