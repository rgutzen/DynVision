# Skip & Feedback Connections

This reference describes DynVision's generalized skip‑ and feedback‑connection
modules and the `auto_adapt` mechanism.

## Description

Skip and feedback connections describe the same process: a copy of a layer's
output is diverted from the immediate feedforward path to be integrated with
a downstream (*skip*) or upstream (*feedback*) region. DynVision provides a
single generalised module that covers both cases and shares the same integration
strategies as recurrent connections (additive / multiplicative / custom callable).

## Auto‑Adapt

Manually matching tensor shapes between source and target layers is tedious and
error‑prone — especially during architecture exploration where layers and
operation order may change frequently.

The `auto_adapt` option solves this: define a connection by referencing a
**source** and **target** layer by name, and the correct up‑/down‑sampling
transform is created automatically on the first forward pass. This lets you
re‑order operations and swap connection targets without worrying about
dimensionality mismatches.

## Biological Motivation

Anatomical studies show that feedback projections from higher cortical areas
(e.g. V4, IT) back to lower areas (V2, V1) are as numerous as feedforward
connections, suggesting a fundamental role in visual computation (Felleman &
Van Essen, 1991; Salin & Bullier, 1995). Skip connections bypass intermediate
processing stages, analogous to long‑range cortico‑cortical projections.

## See Also

- [Layer Operations](layer-operations.md) — where `addskip` and `addfeedback` fit in the pipeline
- [Integration Strategies](integration-strategies.md) — additive / multiplicative / custom
- [Recurrence Types](recurrence-types.md) — lateral recurrent connections
