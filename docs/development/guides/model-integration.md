# Model Integration Guide

> **Purpose**: Translate external PyTorch model implementations into DynVision-conformant versions that leverage the framework's temporal dynamics, recurrence mechanisms, and biological features.

This guide walks through the complete process of integrating a model from the literature (e.g., from a paper's reference implementation) into DynVision's architecture.

---

## Overview

DynVision provides a framework for temporal neural networks with:
- **Recurrent connections** within and between layers
- **Temporal dynamics** with configurable delays and time constants
- **Biological features** like skip connections, feedback, and supralinearity
- **Unified forward pass** via operation sequencing in `TemporalBase`

### When to Use This Guide

- Reimplementing a model from a paper to leverage DynVision's features
- Adding pretrained models (e.g., CorNet, ResNet) to the model zoo
- Creating DynVision-compatible versions of custom architectures
- Ensuring architectural equivalence when loading pretrained weights

---

## Phase 1: Investigation - Understand the Original

**Core Principle**: Never begin implementation before fully understanding the original architecture.

### 1.1 Trace the Complete Forward Pass

Read the original model's forward pass **line by line** and document:

```python
# Example: Original CORblock_RT forward pass
def forward(self, inp=None, state=None, batch_size=None):
    # [1] Feedforward input processing
    inp = self.conv_input(inp)      # Conv2d: channels, kernel, stride
    inp = self.norm_input(inp)      # Normalization type and params
    inp = self.nonlin_input(inp)    # Activation function

    # [2] Recurrence point
    if state is None:
        state = 0
    skip = inp + state              # ← How is state integrated?

    # [3] Second stage
    x = self.conv1(skip)            # Conv2d params
    x = self.norm1(x)               # Normalization
    x = self.nonlin1(x)             # Activation

    # [4] State management
    state = self.output(x)          # What gets stored?
    return output, state
```

**Document for each operation**:
- Input/output shapes at each step
- Parameter values (kernel size, stride, padding, bias, channels)
- Operation order (critical for exact replication)
- Where recurrence is applied (before/after which operations?)
- How state is stored and retrieved

### 1.2 Identify Layer Structure Patterns

Determine the layer organization:

```python
# Original CorNet structure
class CORnet_RT:
    def __init__(self, times=5):
        self.V1 = CORblock_RT(3, 64, kernel_size=7, stride=4)
        self.V2 = CORblock_RT(64, 128, stride=2)
        self.V4 = CORblock_RT(128, 256, stride=2)
        self.IT = CORblock_RT(256, 512, stride=2)
        self.decoder = nn.Sequential(...)
```

**Questions to answer**:
- How many layers/stages?
- Are they similar or heterogeneous?
- What are the channel progressions?
- What are the spatial dimension reductions?
- Is there a pattern that can be generalized?

### 1.3 Analyze Temporal Dynamics

For recurrent models, understand the time loop:

```python
# Original CorNet time loop
for t in range(1, self.times):
    for block in blocks[1:]:
        prev_output = outputs[prev_block]  # Feedforward input
        prev_state = states[block]          # Recurrent state
        new_output, new_state = getattr(self, block)(prev_output, prev_state)
        outputs[block] = new_output
        states[block] = new_state
```

**Document**:
- How many timesteps?
- What initializes at t=0?
- What is passed between timesteps?
- Where does feedforward input come from?
- Where does recurrent input come from?
- What delay is there between storing and retrieving state?

### 1.4 Check Data Preprocessing

Critical for loading pretrained weights:

```python
# Look for data transformations in training scripts
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # ← Check if normalization is applied!
    # transforms.Normalize(mean=[...], std=[...])  # Present or not?
])
```

**Document**:
- Input resolution expected
- Normalization parameters (mean, std) or absence thereof
- Other preprocessing (resizing, cropping, augmentation)
- Expected input range (0-1, 0-255, normalized)

### 1.5 Review Pretrained Weights (if applicable)

If loading pretrained weights:

```python
# Load and inspect
state_dict = torch.load("pretrained.pth")
print(state_dict.keys())

# Check layer names, shapes
for key, value in state_dict.items():
    print(f"{key}: {value.shape}")
```

**Document**:
- Layer naming convention
- Which layers have pretrained weights
- Which layers need to be randomly initialized (e.g., classifier for different n_classes)
- State dict structure (nested? module. prefix?)

---

## Phase 2: Mapping - Find DynVision Equivalents

### 2.1 Choose the Base Class

DynVision provides two main base classes:

| Base Class | Use When |
|------------|----------|
| `BaseModel` | Standard models with minimal temporal dynamics |
| `DyRCNN` | Models with biological features (skip, feedback, retina, tau learning) |

```python
from dynvision.base import BaseModel
# OR
from dynvision.models.dyrcnn import DyRCNN

class MyModel(BaseModel):  # or DyRCNN
    def __init__(self, ...):
        super().__init__(...)
```

**Decision factors**:
- Does the model have recurrent connections? → Consider BaseModel with RConv2d
- Need skip connections between non-adjacent layers? → DyRCNN
- Need feedback connections? → DyRCNN
- Need learnable time constants (tau)? → DyRCNN
- Simple feedforward with optional recurrence? → BaseModel

### 2.2 Map Convolutional Blocks to RConv2d

DynVision's `RConv2d` (RecurrentConnectedConv2d) can represent various block types:

#### **Single-Stage Convolution**

```python
# Original: Simple conv → norm → activation
class Block:
    def __init__(self):
        self.conv = nn.Conv2d(in_c, out_c, k, s, p)
        self.norm = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

# DynVision equivalent
from dynvision.model_components import RConv2d

layer = RConv2d(
    in_channels=in_c,
    out_channels=out_c,
    kernel_size=k,
    stride=s,
    padding=p,
    mid_channels=None,  # ← Single-stage
    recurrence_type="none",  # No recurrence
)
# Then apply norm and activation via layer_operations
```

#### **Two-Stage Convolution** (e.g., CorNet blocks)

```python
# Original: conv1 → norm1 → relu1 → [recurrence] → conv2 → norm2 → relu2
class CORblock:
    def __init__(self):
        self.conv_input = nn.Conv2d(in_c, mid_c, k1, s1, p1, bias=True)
        self.norm_input = nn.GroupNorm(32, mid_c)
        self.nonlin_input = nn.ReLU()
        # [recurrence applied here]
        self.conv1 = nn.Conv2d(mid_c, out_c, k2, s2, p2, bias=False)
        self.norm1 = nn.GroupNorm(32, out_c)
        self.nonlin1 = nn.ReLU()

# DynVision equivalent
layer = RConv2d(
    in_channels=in_c,
    mid_channels=mid_c,      # ← Enables two-stage
    out_channels=out_c,
    kernel_size=(k1, k2),    # ← Tuple for two stages
    stride=(s1, s2),
    bias=(True, False),      # ← Per-stage bias
    mid_modules=nn.GroupNorm(32, mid_c),  # Applied after first conv
    recurrence_target="middle",  # Recurrence between stages
    recurrence_type="self",
    fixed_self_weight=1.0,
    recurrence_bias=False,
)
```

**Key RConv2d parameters**:

| Parameter | Purpose | Values |
|-----------|---------|--------|
| `mid_channels` | Enables two-stage conv | `None` (single) or int (two-stage) |
| `kernel_size` | Kernel size(s) | int or tuple `(k1, k2)` |
| `stride` | Stride(s) | int or tuple `(s1, s2)` |
| `bias` | Bias per stage | bool or tuple `(b1, b2)` |
| `mid_modules` | Operations after first conv | `nn.Module` (e.g., GroupNorm) |
| `recurrence_target` | Where recurrence applies | `"input"`, `"middle"`, `"output"` |
| `recurrence_type` | Type of recurrence | `"none"`, `"self"`, `"full"`, etc. |
| `t_recurrence` | Recurrence delay | float (ms) |
| `fixed_self_weight` | Fixed recurrence weight | float or None (learnable) |

### 2.3 Understand recurrence_target

**Critical**: This determines where the recurrent connection is applied.

```python
# recurrence_target="input"
# Flow: [+ recurrence] → conv → ... → conv2

# recurrence_target="middle" (CorNet case)
# Flow: conv → mid_modules → nonlin → [+ recurrence] → conv2

# recurrence_target="output"
# Flow: conv → ... → conv2 → [+ recurrence]
```

**To determine correct setting**:
1. Trace where `state` or recurrent input is added in original
2. Count operations before/after that point
3. Match to DynVision's structure

### 2.4 Map Normalization and Activation

DynVision separates operations into `layer_operations`:

```python
# Original combined in block
x = conv(x)
x = norm(x)
x = relu(x)

# DynVision: Define operations and let TemporalBase orchestrate
self.layer_operations = [
    "layer",   # Calls RConv2d
    "norm",    # Calls self.norm_LayerName if exists
    "nonlin",  # Calls self.nonlin if exists
    "record",  # Records activations
    "delay",   # Manages hidden state
]

# Define the components
self.LayerName = RConv2d(...)           # The convolution
self.norm_LayerName = nn.GroupNorm(...) # External normalization
self.nonlin = nn.ReLU(inplace=True)     # Shared activation
```

**Decision**: Normalization inside vs. outside RConv2d?

- **Inside** (`mid_modules`): Applied between conv stages in two-stage RConv2d
- **Outside** (`norm_LayerName`): Applied after RConv2d via `layer_operations`

For CorNet, we use both:
- `mid_modules=nn.GroupNorm(...)` for first-stage normalization
- `self.norm_V1 = nn.GroupNorm(...)` for second-stage normalization

### 2.5 Map State Management

DynVision handles state automatically via `ForwardRecurrenceBase`:

```python
# Original manual state management
def forward(self, inp, state=None):
    if state is None:
        state = 0
    x = process(inp)
    x = x + state  # Add recurrent input
    state = x      # Store for next timestep
    return x, state

# DynVision automatic handling
# RConv2d internally:
# - Retrieves: h = self.get_hidden_state(self.delay_recurrence)
# - Processes: h = self.recurrence(h)
# - Integrates: x = self.integrate_signal(x, h)
# - Stores: self.set_hidden_state(x)  # via "delay" operation
```

**Key methods** (you typically don't call these directly):
- `get_hidden_state(delay)`: Retrieve state from `delay` timesteps ago
- `set_hidden_state(x)`: Store current state for future retrieval
- `reset()`: Clear all hidden states (call this at start of forward pass)

---

## Phase 3: Implementation - Build the DynVision Version

### 3.1 Create the Model Class

```python
"""
MyModel: Description of the model and its purpose.

This module implements [model name] from [paper citation], adapted to
DynVision's temporal dynamics framework.

References:
- Author et al. (Year) "Paper Title"
"""

import logging
from typing import Optional, Dict, Any, Tuple, Union, List

import torch
import torch.nn as nn

from dynvision.base import BaseModel  # or DyRCNN
from dynvision.model_components import RConv2d

__all__ = ["MyModel"]

logger = logging.getLogger(__name__)


class MyModel(BaseModel):
    def __init__(
        self,
        # Standard DynVision parameters
        n_timesteps: int = 5,
        input_dims: Tuple[int, int, int] = (5, 3, 224, 224),
        n_classes: int = 1000,
        dt: float = 2,  # ms
        t_feedforward: float = 2,  # ms
        t_recurrence: float = 2,  # ms
        recurrence_type: str = "self",
        # Model-specific parameters
        init_with_pretrained: bool = True,
        **kwargs: Any,
    ) -> None:

        # Store model-specific attributes
        self.init_with_pretrained = init_with_pretrained

        # Pass to parent
        super().__init__(
            n_timesteps=n_timesteps,
            input_dims=input_dims,
            n_classes=n_classes,
            dt=dt,
            t_feedforward=t_feedforward,
            t_recurrence=t_recurrence,
            recurrence_type=recurrence_type,
            **kwargs,
        )
```

### 3.2 Define Architecture in `_define_architecture()`

This is the core method where you define the model structure:

```python
def _define_architecture(self):
    """Define model architecture following DynVision conventions."""

    # [1] Define layer names (must match attribute names!)
    self.layer_names = ["V1", "V2", "V4", "IT"]

    # [2] Define operation sequence
    self.layer_operations = [
        "layer",   # Apply the layer (RConv2d)
        "norm",    # Apply norm_LayerName if it exists
        "nonlin",  # Apply self.nonlin if it exists
        "record",  # Record activations (if store_responses=True)
        "delay",   # Manage hidden states for recurrence
    ]

    # [3] Define shared modules
    self.nonlin = nn.ReLU(inplace=True)

    # [4] Define layers
    self.V1 = RConv2d(
        in_channels=3,
        mid_channels=64,
        out_channels=64,
        kernel_size=(7, 3),
        stride=(4, 1),
        bias=(True, False),
        mid_modules=nn.GroupNorm(32, 64),
        recurrence_type=self.recurrence_type,
        fixed_self_weight=1.0,
        recurrence_bias=False,
        dim_y=self.dim_y,
        dim_x=self.dim_x,
        history_length=int(max(self.t_recurrence, self.t_feedforward) / self.dt) + 1,
    )
    self.norm_V1 = nn.GroupNorm(32, 64)

    # [5] Define classifier
    self.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(512, self.n_classes),
    )
```

**Key points**:
- `layer_names` must exactly match the attribute names you define
- `layer_operations` defines the execution order
- Operations look for attributes named `{operation}_{layer_name}` (e.g., `norm_V1`)
- Or fall back to shared operations (e.g., `self.nonlin`)
- Dimensions (`dim_y`, `dim_x`) are calculated in `_process_input_dimensions()`

### 3.3 Implement `reset()`

Clear hidden states at the start of each forward pass:

```python
def reset(self):
    """Reset all hidden states in recurrent layers."""
    for layer in [self.V1, self.V2, self.V4, self.IT]:
        layer.reset()
```

### 3.4 Handle Pretrained Weights (Optional)

If loading pretrained weights:

```python
def _init_parameters(self) -> None:
    """Initialize or load parameters."""
    if self.init_with_pretrained:
        self.load_pretrained_state_dict(
            check_mismatch_layer=["classifier.2"]  # Layers that may differ
        )
        # Only train classifier if rest is pretrained
        self.trainable_parameter_names = [
            p for p in list(self.state_dict().keys())
            if "classifier.2" in p
        ]
    else:
        # Train all parameters
        self.trainable_parameter_names = list(self.state_dict().keys())

def download_pretrained_state_dict(self):
    """Download pretrained weights."""
    url = "https://example.com/model_weights.pth"
    ckpt_data = torch.utils.model_zoo.load_url(url, map_location=self.device)
    state_dict = ckpt_data["state_dict"]
    # Remove module. prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    return state_dict

def translate_pretrained_layer_names(self):
    """Map original layer names to DynVision names."""
    return {
        # Original name pattern: DynVision name pattern
        "conv_input": "conv",
        "norm_input": "mid_modules",
        "conv1": "conv2",
        "norm1.weight": "norm_LayerName.weight",
        "norm1.bias": "norm_LayerName.bias",
        "decoder.linear": "classifier.2",
    }
```

**Translation process**:
1. `download_pretrained_state_dict()`: Download and load original weights
2. `translate_pretrained_layer_names()`: Define name mappings
3. `load_pretrained_state_dict()`: Automatically applies translation (inherited from BaseModel)
4. `check_mismatch_layer`: Specify layers where shape might differ (e.g., classifier with different n_classes)

### 3.5 Verify Operation Equivalence

**Critical check**: Ensure your DynVision implementation produces the same operation sequence as the original:

```python
# Original sequence (example: CorNet block per timestep)
1. conv_input → norm_input → nonlin_input
2. + state  (recurrence)
3. conv1 → norm1 → nonlin1
4. store state

# Your DynVision sequence (must match!)
# Within RConv2d (recurrence_target="middle"):
1. self.conv → self.mid_modules → self.nonlin
2. + recurrence(h)  where h = get_hidden_state(delay_recurrence)
3. self.conv2
# Via layer_operations:
4. norm_V1  (second normalization)
5. nonlin   (second activation)
6. set_hidden_state(x)  (via "delay" operation)
```

**Method to verify**: Add print statements or use `torch.set_printoptions` and compare intermediate outputs with the original model on the same input.

---

## Phase 4: Validation - Ensure Correctness

### 4.1 Test Model Creation

```python
def test_model_creation():
    """Test that model can be instantiated."""
    model = MyModel(
        input_dims=(5, 3, 224, 224),
        n_classes=1000,
    )
    model.setup("fit")

    assert hasattr(model, 'V1')
    assert hasattr(model, 'classifier')
    print("✓ Model creation successful")
```

### 4.2 Test Forward Pass

```python
def test_forward_pass():
    """Test forward pass with random input."""
    model = MyModel(input_dims=(5, 3, 224, 224))
    model.setup("fit")

    x = torch.randn(2, 5, 3, 224, 224)  # batch=2, timesteps=5
    y = model(x)

    assert y.shape == (2, 5, 1000), f"Expected (2, 5, 1000), got {y.shape}"
    assert torch.isfinite(y).all(), "Output contains NaN or Inf"
    print(f"✓ Forward pass successful: {x.shape} → {y.shape}")
```

### 4.3 Test Pretrained Weight Loading

```python
def test_pretrained_loading():
    """Test loading pretrained weights."""
    model = MyModel(
        input_dims=(5, 3, 224, 224),
        init_with_pretrained=True,
    )
    model.setup("fit")

    # Check that weights were loaded (not random initialization values)
    # You can check specific layer statistics or weight ranges
    v1_weight = model.V1.conv.weight
    assert v1_weight.abs().mean() > 0.001, "Weights appear uninitialized"
    print("✓ Pretrained weights loaded successfully")
```

### 4.4 Compare with Original (Gold Standard)

**Most important validation**: Compare outputs with the original implementation.

```python
def test_equivalence():
    """Compare DynVision implementation with original."""
    from models.original_model import OriginalModel

    # Load same pretrained weights into both
    original = OriginalModel()
    original.load_state_dict(torch.load("pretrained.pth"))
    original.eval()

    dynvision = MyModel(init_with_pretrained=True)
    dynvision.setup("fit")
    dynvision.eval()

    # Same random seed for same input
    torch.manual_seed(42)
    x = torch.randn(1, 5, 3, 224, 224)

    with torch.no_grad():
        y_original = original(x[:, 0])  # If original doesn't handle time
        y_dynvision = dynvision(x)[:, 0]  # First timestep

    # Check outputs are close
    diff = (y_original - y_dynvision).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")

    # Tolerances depend on precision (float32 vs float16)
    assert max_diff < 1e-4, f"Outputs differ too much: {max_diff}"
    print("✓ Outputs match original implementation")
```

**Common reasons for mismatch**:
1. **Wrong operation order**: Check `layer_operations` sequence
2. **Wrong recurrence_target**: Check where `state` is added in original
3. **Missing normalization/activation**: Check all operations are present
4. **Wrong initialization**: Ensure pretrained weights loaded correctly
5. **Numerical precision**: float16 vs float32, check with `dtype` parameter

### 4.5 Test with Different Configurations

```python
def test_configurations():
    """Test various model configurations."""
    configs = [
        {"recurrence_type": "none"},
        {"recurrence_type": "self"},
        {"recurrence_type": "full"},
        {"n_timesteps": 1},
        {"n_timesteps": 10},
        {"feedforward_only": True},
    ]

    for config in configs:
        print(f"Testing config: {config}")
        model = MyModel(
            input_dims=(5, 3, 224, 224),
            **config
        )
        model.setup("fit")

        x = torch.randn(1, 5, 3, 224, 224)
        y = model(x)

        assert torch.isfinite(y).all()
        print(f"  ✓ Passed")
```

---

## Phase 5: Special Considerations

### 5.1 Data Preprocessing Requirements

If the model was trained with specific preprocessing, document and configure it:

```python
# In config_data.yaml, add model requirements:
model_requirements:
    MyModel:
        normalize: null  # If model expects raw pixels (like CorNet)
        # OR
        normalize: [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]  # ImageNet stats
        note: "Model trained without normalization"
```

Then users can override normalization:
```bash
snakemake test_model --config normalize=null model_name=MyModel
```

See the normalization override system implemented in:
- `dynvision/params/data_params.py` (validator handles JSON null)
- `dynvision/workflow/snake_runtime.smk` (passes normalize parameter)

### 5.2 Handling Different Input Sizes

DynVision models should be flexible about input size:

```python
def _define_architecture(self):
    # Use self.dim_y, self.dim_x from parent class
    # These are calculated from input_dims automatically

    self.V1 = RConv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=7,
        stride=4,
        dim_y=self.dim_y,  # ← Passed from input_dims
        dim_x=self.dim_x,  # ← Passed from input_dims
    )

    # Next layer dimensions calculated from previous
    self.V2 = RConv2d(
        in_channels=64,
        out_channels=128,
        dim_y=self.V1.dim_y // self.V1.stride,  # ← Propagate dimensions
        dim_x=self.V1.dim_x // self.V1.stride,
    )
```

If using adaptive pooling, final dimensions are flexible:
```python
self.classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),  # ← Adapts to any spatial size
    nn.Flatten(),
    nn.Linear(512, self.n_classes),
)
```

### 5.3 Handling Multiple Timesteps

DynVision handles time automatically via `TemporalBase.forward()`:

```python
# User code:
x = torch.randn(batch, timesteps, channels, height, width)
y = model(x)  # Shape: (batch, timesteps, n_classes)

# Internally (you don't implement this):
# for t in range(timesteps):
#     x_t = x[:, t]
#     y_t, responses = model._forward(x_t, t=t)
#     # Automatically manages hidden states via layer.set_hidden_state()
```

Your model just needs:
- `reset()`: Clear states at start
- RConv2d layers: Automatically manage hidden states
- Proper `history_length` parameter: Based on max(t_feedforward, t_recurrence, ...)

### 5.4 Skip and Feedback Connections

For advanced models with long-range connections, use DyRCNN:

```python
from dynvision.models.dyrcnn import DyRCNN
from dynvision.model_components import Skip, Feedback

class MyModel(DyRCNN):
    def _define_architecture(self):
        self.layer_names = ["V1", "V2", "V4", "IT"]

        # ... define layers ...

        # Skip connection: V1 → V4 (bypass V2)
        if self.skip:
            self.addskip_V4 = Skip(
                source=self.V1,
                auto_adapt=True,  # Automatically handles shape mismatch
                delay_index=self.t_skip // self.dt,
            )

        # Feedback connection: V4 → V1
        if self.feedback:
            self.addfeedback_V1 = Feedback(
                source=self.V4,
                auto_adapt=True,
                delay_index=self.t_feedback // self.dt,
            )
```

The `auto_adapt=True` automatically learns the transformation to match shapes via a simple linear projection.

### 5.5 Common Pitfalls and Solutions

#### Pitfall 1: Wrong recurrence_target

**Symptom**: Model loads weights but outputs don't match original.

**Solution**: Carefully trace where recurrence is applied in original:
- Before first operation? → `recurrence_target="input"`
- Between two-stage convolutions? → `recurrence_target="middle"`
- After all operations? → `recurrence_target="output"`

#### Pitfall 2: Missing operations after RConv2d

**Symptom**: Pretrained weights load but accuracy is poor.

**Solution**: Check if operations like normalization/activation are INSIDE or OUTSIDE RConv2d:

```python
# Original has TWO normalizations:
# 1. After first conv (inside mid_modules)
# 2. After second conv (outside, via layer_operations)

# DynVision needs both:
layer = RConv2d(
    mid_modules=nn.GroupNorm(32, 64),  # First norm
    ...
)
self.norm_LayerName = nn.GroupNorm(32, 64)  # Second norm
```

#### Pitfall 3: Dimension mismatches

**Symptom**: Runtime error about tensor shape mismatch.

**Solution**:
1. Print shapes at each step during initialization
2. Ensure `dim_y`, `dim_x` are passed correctly
3. Account for all striding and pooling when calculating next layer dims

```python
# Debug dimension propagation
print(f"Input: {self.dim_y} x {self.dim_x}")
print(f"After V1 (stride={self.V1.stride}): {self.V1.dim_y} x {self.V1.dim_x}")
```

#### Pitfall 4: State not persisting across timesteps

**Symptom**: Recurrent connections don't seem to work, behavior same as feedforward.

**Solution**:
1. Ensure `reset()` is called (happens automatically in `TemporalBase.forward()`)
2. Check `history_length` is sufficient: `max(t_recurrence, t_feedforward) / dt + 1`
3. Verify `delay_recurrence` matches original model's delay
4. Ensure "delay" operation is in `layer_operations`

---

## Phase 6: Documentation and Testing

### 6.1 Write Docstrings

```python
class MyModel(BaseModel):
    """
    MyModel: [One-line description]

    [Detailed description of what this model implements, its purpose,
    and key features]

    This implementation adapts [Original Model] from [Citation] to
    DynVision's temporal framework, enabling:
    - Recurrent processing with configurable delays
    - Temporal dynamics with learnable time constants
    - Integration with DynVision's training and evaluation pipeline

    Architecture:
        - Layer 1: [Description]
        - Layer 2: [Description]
        - ...

    Args:
        n_timesteps: Number of timesteps to process
        input_dims: Input dimensions (timesteps, channels, height, width)
        n_classes: Number of output classes
        dt: Integration time step (ms)
        t_recurrence: Recurrent connection delay (ms)
        recurrence_type: Type of recurrent connections ("self", "full", "none")
        init_with_pretrained: Whether to load pretrained weights

    References:
        - Author et al. (Year) "Paper Title" Journal/Conference
        - Original implementation: [URL if available]

    Example:
        >>> model = MyModel(
        ...     input_dims=(5, 3, 224, 224),
        ...     n_classes=1000,
        ...     init_with_pretrained=True,
        ... )
        >>> model.setup("fit")
        >>> x = torch.randn(2, 5, 3, 224, 224)
        >>> y = model(x)  # Shape: (2, 5, 1000)
    """
```

### 6.2 Create Test Function

```python
def test_mymodel(
    input_shape: Tuple[int, ...] = (5, 3, 224, 224),
    device: Optional[torch.device] = None,
) -> None:
    """Test MyModel implementation.

    Tests include:
    - Model creation and setup
    - Forward pass with random input
    - Stability checks with extreme values
    - Parameter verification

    Args:
        input_shape: Input tensor shape
        device: Device to run test on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing MyModel on {device}")

    # Create and setup model
    model = MyModel(input_dims=input_shape)
    model.setup("fit")
    print("✓ Model creation successful")

    # Test forward pass
    x = torch.randn(1, *input_shape, device=device)
    y = model(x)
    print(f"✓ Forward pass successful: {x.shape} -> {y.shape}")

    # Test with pretrained weights
    if model.init_with_pretrained:
        try:
            model = MyModel(input_dims=input_shape, init_with_pretrained=True)
            model.setup("fit")
            print("✓ Pretrained weights loaded successfully")
        except Exception as e:
            print(f"⚠ Pretrained loading failed: {e}")

    # Log trainable parameters
    trainable_params = [
        f"{name} [{tuple(param.shape)}]"
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    print("Trainable Parameters:\n\t%s" % "\n\t".join(trainable_params))

    print("All tests passed!")


if __name__ == "__main__":
    test_mymodel()
```

### 6.3 Add to Model Zoo

1. **Import in `__init__.py`**:
```python
# dynvision/models/__init__.py
from .mymodel import MyModel

__all__ = [..., "MyModel"]
```

2. **Update documentation**:
```markdown
# docs/reference/models.md

## MyModel

[Description]

**Usage:**
\`\`\`python
from dynvision.models import MyModel

model = MyModel(
    input_dims=(5, 3, 224, 224),
    n_classes=1000,
    init_with_pretrained=True,
)
\`\`\`

**Parameters:**
- ...

**References:**
- ...
```

---

## Quick Reference: DynVision Architecture Patterns

### Pattern 1: Simple Feedforward with Optional Recurrence

```python
class SimpleModel(BaseModel):
    def _define_architecture(self):
        self.layer_names = ["conv1", "conv2"]
        self.layer_operations = ["layer", "nonlin", "pool", "record", "delay"]

        self.conv1 = RConv2d(3, 64, 3, stride=1, recurrence_type="self")
        self.conv2 = RConv2d(64, 128, 3, stride=2, recurrence_type="self")
        self.nonlin = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Linear(128, 10)
```

### Pattern 2: Two-Stage Blocks (CorNet-style)

```python
class TwoStageModel(BaseModel):
    def _define_architecture(self):
        self.layer_names = ["block1"]
        self.layer_operations = ["layer", "norm", "nonlin", "record", "delay"]

        self.block1 = RConv2d(
            in_channels=3,
            mid_channels=64,
            out_channels=64,
            kernel_size=(7, 3),
            stride=(4, 1),
            mid_modules=nn.GroupNorm(32, 64),
            recurrence_target="middle",
        )
        self.norm_block1 = nn.GroupNorm(32, 64)
        self.nonlin = nn.ReLU()
```

### Pattern 3: Biological Features (DyRCNN-style)

```python
class BiologicalModel(DyRCNN):
    def _define_architecture(self):
        self.layer_names = ["V1", "V4"]
        self.layer_operations = [
            "layer", "addskip", "addfeedback", "tstep",
            "nonlin", "record", "delay"
        ]

        self.V1 = RConv2d(3, 64, 3, recurrence_type="full", **params)
        self.tstep_V1 = EulerStep(dt=self.dt, tau=self.tau)

        self.V4 = RConv2d(64, 128, 3, **params)
        self.tstep_V4 = EulerStep(dt=self.dt, tau=self.tau)

        if self.skip:
            self.addskip_V4 = Skip(source=self.V1, auto_adapt=True)
        if self.feedback:
            self.addfeedback_V1 = Feedback(source=self.V4, auto_adapt=True)
```

---

## Summary Checklist

Before considering integration complete:

**Investigation Phase**:
- [ ] Traced original forward pass line by line
- [ ] Documented all layer parameters (channels, kernels, strides, bias)
- [ ] Identified where recurrence is applied
- [ ] Checked data preprocessing requirements
- [ ] Inspected pretrained weights structure (if applicable)

**Mapping Phase**:
- [ ] Chose appropriate base class (BaseModel vs DyRCNN)
- [ ] Mapped convolutional blocks to RConv2d parameters
- [ ] Determined correct `recurrence_target` value
- [ ] Identified operations inside vs. outside RConv2d
- [ ] Planned `layer_operations` sequence

**Implementation Phase**:
- [ ] Defined `layer_names` matching attribute names
- [ ] Implemented `_define_architecture()` with all layers
- [ ] Implemented `reset()` for hidden state management
- [ ] Added pretrained weight loading (if applicable)
- [ ] Created layer name translation mapping (if applicable)

**Validation Phase**:
- [ ] Tested model creation and initialization
- [ ] Tested forward pass with various inputs
- [ ] Compared outputs with original implementation
- [ ] Tested with different configurations
- [ ] Verified pretrained weights load correctly

**Documentation Phase**:
- [ ] Written comprehensive docstrings
- [ ] Created test function
- [ ] Added to model zoo (`__init__.py`)
- [ ] Updated documentation
- [ ] Noted any special preprocessing requirements

---

## Further Reading

- **DynVision Architecture**: [`docs/development/guides/claude-guide.md`](claude-guide.md)
- **AI Development Guide**: [`docs/development/guides/ai-style-guide.md`](ai-style-guide.md)
- **Base Classes**: [`dynvision/base/temporal.py`](../../dynvision/base/temporal.py)
- **Recurrence Components**: [`dynvision/model_components/recurrence.py`](../../dynvision/model_components/recurrence.py)
- **Example Models**:
  - Simple: [`dynvision/models/cornet_rt.py`](../../dynvision/models/cornet_rt.py)
  - Complex: [`dynvision/models/dyrcnn.py`](../../dynvision/models/dyrcnn.py)

---

*"Before building new, understand what exists. Before coding, trace the original. Before claiming equivalence, validate with data."*
