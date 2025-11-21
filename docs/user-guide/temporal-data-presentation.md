# Temporal Data Presentation

This guide explains how DynVision handles temporal data presentation across different components of the system, from data loading to model processing.

## Overview

DynVision provides multiple mechanisms for creating and managing temporal sequences:

1. **DataLoader-based expansion** - Expands static images into temporal sequences during loading (for testing)
2. **FFCV-based expansion** - High-performance temporal expansion using FFCV pipelines (for training)
3. **Model-based expansion** - Dynamic temporal expansion within the model's forward pass (most flexible)

Each approach has different use cases and performance characteristics.

---

## Configuration Parameters

### Key Parameters

**`data_timesteps` (alias: `dsteps`):**
- Controls temporal expansion in **dataloaders**
- Applied during data loading, before model receives data
- Used by both standard PyTorch dataloaders and FFCV loaders
- Default: `1` (no temporal expansion)

**`n_timesteps` (alias: `tsteps`):**
- Controls temporal expansion in **model forward pass**
- Applied dynamically within TemporalBase._expand_timesteps()
- Allows presentation patterns and shuffling
- Default: `1` (no expansion)

**Configuration location:** `dynvision/configs/config_data.yaml`

```yaml
data_timesteps: 1  # DataLoader expansion
```

**Important:** Only one temporal expansion method should be active at a time:
- For **training** with FFCV: Set `data_timesteps > 1`, keep model `n_timesteps = 1`
- For **testing/flexibility**: Set `data_timesteps = 1`, use model `n_timesteps > 1` with patterns
- Never set both > 1 (results in double expansion)

---

## Temporal Expansion Methods

### 1. DataLoader-Based Expansion (Testing)

**Purpose:** Simple temporal expansion for testing and prototyping

**Location:** `dynvision/data/dataloader.py`

**Available DataLoaders:**

#### StandardDataLoader
Basic temporal repetition without void periods.

```python
from dynvision.data.dataloader import StandardDataLoader

loader = StandardDataLoader(
    dataset,
    n_timesteps=20,      # Repeat each image 20 times
    batch_size=32,
    num_workers=4,
)

# Output shape: [batch, 20, channels, height, width]
```

#### StimulusDurationDataLoader
Adds intro and outro void periods around stimulus.

```python
from dynvision.data.dataloader import StimulusDurationDataLoader

loader = StimulusDurationDataLoader(
    dataset,
    n_timesteps=30,           # Total sequence length
    stimulus_duration=20,     # Stimulus presentation duration
    intro_duration=5,         # Void period before stimulus
    non_input_value=0.0,      # Value for void timesteps
    non_label_index=-1,       # Label for void timesteps
)

# Timeline: [5 void] + [20 stimulus] + [5 void]
```

**Parameters:**
- `n_timesteps`: Total sequence length
- `stimulus_duration`: How long stimulus is shown
- `intro_duration`: Void timesteps before stimulus
- `outro_duration`: Automatically calculated as `n_timesteps - stimulus - intro`
- `non_input_value`: Input value during void periods (typically 0.0)
- `non_label_index`: Target label during void periods (typically -1)

#### StimulusIntervalDataLoader
Two stimulus presentations separated by an interval.

```python
loader = StimulusIntervalDataLoader(
    dataset,
    n_timesteps=30,
    stimulus_duration=8,      # Each presentation duration
    interval_duration=6,      # Gap between presentations
    intro_duration=2,
)

# Timeline: [2 intro] + [8 stim1] + [6 interval] + [8 stim2] + [6 outro]
```

#### StimulusNoiseDataLoader
Adds noise to stimulus with temporal control.

```python
loader = StimulusNoiseDataLoader(
    dataset,
    n_timesteps=20,
    stimulus_duration=15,
    noise_type="gaussian",    # gaussian, uniform, saltpepper, poisson
    ssnr=0.7,                 # Signal-to-signal+noise ratio
    temporal_mode="static",   # static, dynamic, correlated
    noise_void=True,          # Apply noise to void periods
)
```

**Temporal Noise Modes:**
- `static`: Same noise pattern repeated across timesteps
- `dynamic`: Independent noise per timestep
- `correlated`: Temporally correlated noise

**Performance Features:**
- JIT-compiled tensor operations for speed
- Pre-allocated tensor caching with LRU eviction
- CUDA stream support for async operations
- Channels-last memory format for GPU efficiency

**When to Use:**
- Quick prototyping and testing
- Small-scale experiments
- Situations requiring custom temporal patterns
- When FFCV is unavailable

---

### 2. FFCV-Based Expansion (Training)

**Purpose:** High-performance temporal expansion for large-scale training

**Location:** `dynvision/data/ffcv_dataloader.py`

**Features:**
- ~10-100x faster than standard PyTorch DataLoader
- Optimized memory access patterns
- Built-in GPU transfer
- Minimal Python overhead

**Usage:**

```python
from dynvision.data.ffcv_dataloader import get_ffcv_dataloader

loader = get_ffcv_dataloader(
    path="path/to/dataset.beton",
    batch_size=256,
    data_timesteps=20,        # Temporal expansion
    num_workers=8,
    device=torch.device("cuda:0"),
    dtype=torch.float16,      # Mixed precision
    encoding="image",
    resolution=224,
)

# Output: [batch, 20, 3, 224, 224] directly on GPU
```

**Pipeline Operations:**

1. **Image Decoding** - `RandomResizedCropRGBImageDecoder` or `NDArrayDecoder`
2. **Transforms** - Data augmentation (optional)
3. **Normalization** - Dataset-specific mean/std (optional)
4. **Type Conversion** - `ToTensor()`, `ToTorchImage()`, `Convert(dtype)`
5. **Device Transfer** - `ToDevice(device)`
6. **Temporal Extension** - `ExtendDataTimeFFCV(n_timesteps)` and `ExtendLabelTimeFFCV(n_timesteps)`

**Transform Configuration:**

Transforms are configured in `config_data.yaml`:

```yaml
transform_presets:
  ffcv:
    train:
      base:
        - "RandomHorizontalFlip()"
        - "RandomBrightness(0.2)"
        - "RandomContrast(0.2)"
        - "RandomSaturation(0.2)"
        - "RandomTranslate(padding=22, fill=(0, 0, 0))"
```

**When to Use:**
- Large-scale training (ImageNet, etc.)
- Maximum throughput required
- GPU training with mixed precision
- Production training pipelines

**Limitations:**
- Requires pre-processed .beton files
- No dynamic presentation patterns
- Fixed temporal expansion (all timesteps identical)

---

### 3. Model-Based Expansion (Flexible)

**Purpose:** Dynamic temporal expansion with presentation patterns and shuffling

**Location:** `dynvision/base/temporal.py` - `TemporalBase._expand_timesteps()`

**Features:**
- Presentation patterns (stimulus/null sequences)
- Per-batch pattern shuffling
- Reaction time masking
- Residual timestep handling
- Full control over temporal dynamics

**Basic Usage:**

```python
from dynvision.models import DyRCNNx4

model = DyRCNNx4(
    n_classes=10,
    input_dims=(20, 3, 64, 64),  # (n_timesteps, channels, height, width)
    n_timesteps=20,               # Model handles expansion
    dt=2.0,                       # Time step duration (ms)
)

# DataLoader provides static images: [batch, 1, 3, 64, 64]
# Model expands to: [batch, 20, 3, 64, 64]
```

**Presentation Patterns:**

Control which timesteps receive stimulus vs. null input:

```python
model = DyRCNNx4(
    n_timesteps=10,
    data_presentation_pattern="1011111101",  # Pattern string
    # Or equivalently:
    # data_presentation_pattern=[1, 0, 1, 1, 1, 1, 1, 1, 0, 1]
)

# Timeline interpretation:
# Timestep:  0  1  2  3  4  5  6  7  8  9
# Pattern:   1  0  1  1  1  1  1  1  0  1
# Meaning:   S  N  S  S  S  S  S  S  N  S
# (S=stimulus, N=null/zero input)
```

**Pattern Specification:**
- `"1"` or `[1]`: All timesteps receive stimulus (default)
- `"1011"`: Custom pattern (1=stimulus, 0=null)
- `[1, 0, 1, 1]`: List format (same as string)
- Length must divide evenly into `n_timesteps` (pattern repeats if needed)

**Pattern Resampling:**

Automatically resamples pattern to match `n_timesteps`:

```python
# Pattern: "101" (length 3), n_timesteps=9
# Resampled: "101101101" (repeats 3 times)

# Pattern: "1001" (length 4), n_timesteps=8
# Resampled: "10011001" (repeats 2 times)

# Pattern: "10" (length 2), n_timesteps=7
# Error! 7 is not evenly divisible by 2
```

**Pattern Shuffling:**

Randomly shuffle presentation order per batch:

```python
model = DyRCNNx4(
    n_timesteps=12,
    data_presentation_pattern="100111",  # Base pattern (length 6)
    shuffle_presentation_pattern=True,
)

# Original pattern:  100111 100111
# Shuffled example:  111001 110001
# Each batch gets different random permutation of pattern chunks
```

**Shuffling Behavior:**
- Shuffles the base pattern entries **before** resampling to `n_timesteps`
- Each pattern chunk (e.g., "100111") maintains its duration after shuffle
- Different random order per batch
- Ensures temporal variability during training

**Null Input Handling:**

Timesteps with pattern value `0` receive:
- **Input:** Zero tensor (`torch.zeros_like(input)`)
- **Label:** `non_label_index` (default -1, ignored by loss)

This allows the model to process null periods without supervision.

---

## Reaction Time Masking

**Purpose:** Mask labels immediately after stimulus onset to account for neural processing delays

**Parameter:** `loss_reaction_time` (alias: `lossrt`) in milliseconds

**Behavior:**

```python
model = DyRCNNx4(
    n_timesteps=10,
    dt=5.0,                      # 5ms per timestep
    loss_reaction_time=12.0,     # 12ms reaction window
    data_presentation_pattern="0011111000",
)

# Reaction steps: ceil(12 / 5) = 3 timesteps
# Pattern:  0  0  1  1  1  1  1  0  0  0
# Chunk:    [N][N][  Stimulus 1     ][N][N]
# Masked:         ^  ^  ^  (first 3 of stimulus)
#
# Labels at timesteps 2, 3, 4 are set to non_label_index
```

**Per-Chunk Masking:**

Reaction time applies to **every stimulus chunk** (rising edge in pattern):

```python
model = DyRCNNx4(
    n_timesteps=20,
    dt=2.0,
    loss_reaction_time=6.0,      # 6ms = 3 timesteps
    data_presentation_pattern="00111100001111000011",
)

# reaction_steps = ceil(6 / 2) = 3
#
# Timestep: 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
# Pattern:  0  0  1  1  1  1  0  0  0  0  1  1  1  1  0  0  0  0  1  1
# Chunks:   [null][  chunk1  ][  null   ][  chunk2  ][  null   ][ch3]
# Masked:         ^  ^  ^                 ^  ^  ^                 ^  ^
#
# Masked timesteps: 2, 3, 4, 10, 11, 12, 18, 19
```

**Warnings:**

If reaction window exceeds chunk duration, a warning is logged:

```python
# WARNING: Reaction time (10ms) exceeds chunk duration (6ms)
# at timestep 2. Entire chunk will be masked.
```

**Implementation:** See [Loss Functions Reference](../reference/losses.md#temporal-masking-and-presentation-patterns)

---

## Residual Timesteps

**Purpose:** Handle mismatch between data temporal dimension and model configuration

**Automatic Calculation:**

```python
model = DyRCNNx4(
    n_timesteps=20,
    input_dims=(17, 3, 64, 64),  # Data has 17 timesteps
)

# Residual: 20 - 17 = 3
# Model expects 20, but receives 17
# n_residual_timesteps = 3
```

**Behavior:**

The `n_residual_timesteps` attribute tracks this mismatch:

```python
print(model.n_residual_timesteps)  # Output: 3
```

**Use Cases:**

1. **Data-Model Alignment Tracking** - Monitors temporal dimension consistency
2. **Legacy Compatibility** - Previously used for reaction time calculation
3. **Debugging** - Helps identify configuration mismatches

**Current Status:**

Residual timesteps are **tracked but not actively used** for most operations. Temporal expansion in `_expand_timesteps()` handles dimension matching automatically.

**Historical Note:**

Previously used in fixed-prefix label masking (deprecated). Now replaced by pattern-aware reaction time masking.

---

## Choosing the Right Method

### Decision Matrix

| Use Case | Method | Parameters | Advantages |
|----------|--------|------------|------------|
| **Large-scale training** | FFCV | `data_timesteps > 1` | 10-100x faster, optimized GPU transfer |
| **Flexible patterns** | Model-based | `n_timesteps > 1`, patterns | Dynamic patterns, shuffling, reaction masking |
| **Quick testing** | DataLoader | `n_timesteps > 1` | Simple, no preprocessing required |
| **Custom temporal structures** | DataLoader | Specific loader class | Interval, noise, contrast experiments |
| **Production inference** | Model-based | `n_timesteps > 1` | Consistent with training, flexible |

### Typical Workflows

**Training Workflow (Recommended):**

```python
# 1. Prepare FFCV dataset with data_timesteps
# config_data.yaml:
# data_timesteps: 20

# 2. Create FFCV loader
train_loader = get_ffcv_dataloader(
    path="imagenet_train.beton",
    batch_size=256,
    data_timesteps=20,  # Temporal expansion in loader
)

# 3. Model expects pre-expanded data
model = DyRCNNx4(
    n_timesteps=1,  # No model expansion needed
    input_dims=(20, 3, 224, 224),  # Matches loader output
)
```

**Testing Workflow (Pattern-based):**

```python
# 1. Standard dataloader with static images
test_loader = DataLoader(dataset, batch_size=32)

# 2. Model handles all temporal expansion
model = DyRCNNx4(
    n_timesteps=20,
    data_presentation_pattern="1011111101",
    shuffle_presentation_pattern=False,  # Deterministic for testing
    loss_reaction_time=4.0,
)

# Data: [batch, 1, 3, 224, 224]
# Model expands to: [batch, 20, 3, 224, 224]
```

**Prototyping Workflow (DataLoader):**

```python
# Use custom dataloader for specific temporal structure
loader = StimulusDurationDataLoader(
    dataset,
    n_timesteps=30,
    stimulus_duration=20,
    intro_duration=5,
)

# Model receives pre-expanded data
model = DyRCNNx4(
    n_timesteps=1,
    input_dims=(30, 3, 64, 64),
)
```

---

## Implementation Details

### Expansion in TemporalBase

**Location:** `dynvision/base/temporal.py:_expand_timesteps()`

**Process:**

1. **Check if expansion needed:**
   ```python
   if inputs.size(1) == 1 and self.n_timesteps > 1:
       # Expand from [batch, 1, C, H, W] to [batch, n_timesteps, C, H, W]
   ```

2. **Get presentation pattern:**
   ```python
   presentation_pattern = self._get_presentation_pattern()
   # Returns boolean tensor of length n_timesteps
   ```

3. **Pattern shuffling (if enabled):**
   ```python
   if self.shuffle_presentation_pattern:
       # Shuffle base pattern before resampling
       pattern = permute_pattern_chunks(pattern)
   ```

4. **Compute reaction mask:**
   ```python
   reaction_mask = self._compute_reaction_mask(presentation_pattern)
   # Marks first N timesteps after each stimulus onset
   ```

5. **Apply masks:**
   ```python
   # Clone tensors (required for in-place modification)
   inputs = inputs.clone()
   label_indices = label_indices.clone()

   # Zero out null input timesteps
   inputs[:, zero_mask] = 0

   # Mask null + reaction timesteps in labels
   combined_mask = zero_mask | reaction_mask
   label_indices[:, combined_mask] = self.non_label_index
   ```

**Performance Optimizations:**

- Fully vectorized using PyTorch broadcasting
- Zero GPU-CPU synchronization
- Pattern detection uses tensor operations only
- Supports channels-last memory format
- Automatic device handling for GPU/CPU

### Expansion in DataLoaders

**StandardDataLoader Process:**

```python
def __iter__(self):
    for sample in super().__iter__():
        data, labels, *extra = sample

        # Adjust dimensions
        data = _adjust_data_dimensions(data)
        labels = _adjust_label_dimensions(labels)

        # Temporal expansion
        if self.n_timesteps > 1:
            data = _repeat_over_time(data, self.n_timesteps)
            labels = _repeat_over_time(labels, self.n_timesteps)

        yield [data, labels, *extra]
```

**StimulusDurationDataLoader Process:**

```python
def __iter__(self):
    for sample in DataLoader.__iter__(self):
        data, labels, *extra = sample

        # Get pre-allocated output tensors (cached)
        output_data, output_labels = self._get_cached_tensors(
            data.shape, labels.shape, data.device, data.dtype
        )

        # Pre-fill with void values
        output_data.fill_(self.non_input_value)
        output_labels.fill_(self.non_label_index)

        # Fill stimulus period using JIT-compiled function
        time_idx = self.intro_duration
        expanded_data = self._expand_jit(data, 1, self.stimulus_duration)

        self._fill_period_jit(
            output_data, output_labels,
            expanded_data, expanded_labels,
            time_idx, self.stimulus_duration
        )

        yield [output_data, output_labels, *extra]
```

**FFCV Pipeline:**

```python
# ExtendDataTimeFFCV operation
class ExtendDataTimeFFCV(Operation):
    def generate_code(self):
        # Expand: [batch, C, H, W] -> [batch, n_timesteps, C, H, W]
        return """
        output = input.unsqueeze(1).expand(-1, {n_timesteps}, -1, -1, -1)
        """.format(n_timesteps=self.n_timesteps)
```

---

## Common Patterns

### Pattern 1: Continuous Stimulus Presentation

```python
# All timesteps receive stimulus
data_presentation_pattern="1111111111"
# Or simply: "1" (auto-expanded)
```

**Use Case:** Standard temporal processing without null periods

### Pattern 2: Brief Stimulus with Null Periods

```python
# Intro (2) + Stimulus (6) + Outro (2)
data_presentation_pattern="0011111100"
```

**Use Case:** Testing transient responses, measuring decay

### Pattern 3: Alternating Stimulus-Null

```python
# Alternating pattern
data_presentation_pattern="1010101010"
```

**Use Case:** Studying adaptation, temporal integration

### Pattern 4: Multiple Brief Presentations

```python
# Three brief presentations with gaps
data_presentation_pattern="001100110011"
```

**Use Case:** Sequence learning, working memory experiments

### Pattern 5: Long Stimulus with Interruption

```python
# Long presentation interrupted by null
data_presentation_pattern="111111110011111111"
```

**Use Case:** Studying persistence, attention

---

## Best Practices

### Performance Optimization

1. **Use FFCV for training:**
   ```python
   # Fastest option for large-scale training
   loader = get_ffcv_dataloader(..., data_timesteps=20)
   model = DyRCNNx4(n_timesteps=1, input_dims=(20, ...))
   ```

2. **Enable caching in DataLoaders:**
   ```python
   loader = StimulusDurationDataLoader(
       ...,
       max_cache_size=100,  # Cache pre-allocated tensors
   )
   ```

3. **Use channels-last for GPU:**
   ```python
   loader = StandardDataLoader(
       ...,
       use_channels_last=True,  # Better GPU performance
   )
   ```

4. **Enable CUDA streams:**
   ```python
   loader = StandardDataLoader(
       ...,
       use_cuda_streams=True,  # Overlap computation
   )
   ```

### Configuration Consistency

1. **Match temporal dimensions:**
   ```python
   # DataLoader output: [batch, 20, 3, 224, 224]
   # Model input_dims:   (20, 3, 224, 224)  ✓ Matches
   ```

2. **Avoid double expansion:**
   ```python
   # BAD: Both expand to 20 timesteps
   loader = get_ffcv_dataloader(..., data_timesteps=20)
   model = DyRCNNx4(n_timesteps=20)  # Result: 400 timesteps!

   # GOOD: Expansion in one place only
   loader = get_ffcv_dataloader(..., data_timesteps=20)
   model = DyRCNNx4(n_timesteps=1)
   ```

3. **Pattern length compatibility:**
   ```python
   # GOOD: 20 is divisible by 4
   DyRCNNx4(n_timesteps=20, data_presentation_pattern="1011")

   # BAD: 20 is not divisible by 3
   DyRCNNx4(n_timesteps=20, data_presentation_pattern="101")  # Error!
   ```

### Testing and Debugging

1. **Start simple:**
   ```python
   # Test without patterns first
   model = DyRCNNx4(n_timesteps=10, data_presentation_pattern="1")
   ```

2. **Verify shapes:**
   ```python
   for batch in loader:
       data, labels = batch[:2]
       print(f"Data: {data.shape}, Labels: {labels.shape}")
       break
   ```

3. **Check masking:**
   ```python
   # Verify labels are masked correctly
   print(f"Masked labels: {(labels == -1).sum()} / {labels.numel()}")
   ```

4. **Monitor warnings:**
   ```python
   # Watch for reaction time warnings
   # WARNING: Reaction time (10ms) exceeds chunk duration (6ms)
   ```

---

## Troubleshooting

### Issue: Shape Mismatch Error

**Error:** `RuntimeError: Expected input with shape [batch, 20, ...] but got [batch, 1, ...]`

**Cause:** Model expects temporal expansion but dataloader isn't providing it

**Solution:**
```python
# Either expand in dataloader:
loader = StandardDataLoader(..., n_timesteps=20)
model = DyRCNNx4(n_timesteps=1, input_dims=(20, ...))

# Or expand in model:
loader = DataLoader(dataset)  # Static images
model = DyRCNNx4(n_timesteps=20)
```

### Issue: Pattern Length Error

**Error:** `ValueError: n_timesteps (25) must be evenly divisible by pattern length (7)`

**Cause:** Pattern length doesn't divide evenly into n_timesteps

**Solution:**
```python
# Choose compatible values:
DyRCNNx4(n_timesteps=21, data_presentation_pattern="1011011")  # 21 / 7 = 3 ✓
```

### Issue: All Labels Masked

**Warning:** `All labels are invalid for this batch`

**Cause:** Reaction time masks all timesteps, or pattern has only null inputs

**Solutions:**
```python
# 1. Reduce reaction time
model.loss_reaction_time = 2.0  # Instead of 10.0

# 2. Increase stimulus duration
model.data_presentation_pattern = "0111111100"  # Longer stimulus

# 3. Check pattern validity
# Pattern: "0000" - all null, no valid labels!
```

### Issue: Poor FFCV Performance

**Symptoms:** FFCV loader slower than expected

**Solutions:**
```python
# 1. Increase workers
loader = get_ffcv_dataloader(..., num_workers=16)

# 2. Adjust batches_ahead
loader = get_ffcv_dataloader(..., batches_ahead=4)

# 3. Enable OS cache
loader = get_ffcv_dataloader(..., os_cache=True)

# 4. Verify .beton file location (should be on SSD)
```

---

## Related Documentation

- [Loss Functions Reference](../reference/losses.md) - Loss normalization with temporal masking
- [Model Base Classes](../reference/model-base.md) - TemporalBase implementation details
- [Configuration System](configuration.md) - YAML configuration reference
- [FFCV Integration](../tutorials/ffcv-setup.md) - Setting up FFCV dataloaders
