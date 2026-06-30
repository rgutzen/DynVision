# Layer Operations

This reference describes the configurable order of operations within each network
layer and the reserved operation names.

## Description

DynVision models define each layer as a sequence of operations executed in a
user‑configurable order (the `layer_operations` list). This design enables
researchers to control precisely where recurrent, skip, and feedback signals are
integrated relative to feedforward convolutions and the dynamics solver without
modifying model code.

## Default Operation Order (DyRCNN)

```
layer_operations:
    - "rconv"        # apply recurrent convolutional module
    - "addskip"      # add activity from skip connections
    - "addfeedback"  # add activity from feedback connections
    - "tstep"        # apply dynamical systems ODE solver step
    - "nonlin"       # apply nonlinearity
    - "record"       # record activations in storage buffer
    - "delay"        # set and get delayed activations
    - "pool"         # apply pooling
```

<p align="center">
  <img src="../assets/rcnn_architecture.png" alt="Layer Operations and Architecture" width="700"/>
</p>

*Figure: Architecture schematic of the DyRCNN model family. The left side shows
the signal flow within one layer; the right side shows layer parameters and
cross‑layer connections. Feedforward, recurrent, skip, and feedback delays are
indicated by pointers into hidden‑state timestep slots.*

## Reserved Operation Names

| Name      | Behaviour |
|-----------|-----------|
| `delay`   | Writes the current activity into the hidden states and retrieves past activity with the correct delay. |
| `tstep`   | Applies the numerical ODE solver step (accesses the most recent hidden state). |
| `record`  | Stores the current activity in the model's StorageBuffer. |

## Adding Custom Operations

Each operation corresponds to a method or module attribute on the model. For
layer‑specific operations the naming convention is
`self.<operation>_<layer_name> = <module>`. For layer‑unspecific operations
(e.g. a nonlinearity) the `_<layer_name>` suffix can be omitted.

During the forward pass `TemporalBase` iterates over the `layer_operations`
list and calls each operation for each layer. Operations not defined for a
given layer are silently skipped — heterogeneous layer configurations are
supported without extra code paths.

## See Also

- [Dynamics Solvers](dynamics-solvers.md)
- [Skip & Feedback Connections](skip-feedback-connections.md)
- [Integration Strategies](integration-strategies.md)
