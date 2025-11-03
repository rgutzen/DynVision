# Model Integration: Lessons Learned from Practice

> **Purpose**: This document distills critical lessons learned from reimplementing models (particularly CorNet-RT) into DynVision, focusing on framework-specific considerations and common pitfalls.

This guide complements the main [Model Integration Guide](model-integration.md) by providing practical wisdom gained through real implementation experience.

---

## Quick Start: Step-by-Step Summary

For rapid reference, here's the complete integration workflow:

### Phase 1: Investigation (Don't Skip!)
1. **Obtain original code** and pretrained weights
2. **Trace forward pass** line-by-line, document every operation
3. **Map temporal structure**: Which layers, what delays, where is recurrence applied?
4. **Check data preprocessing**: Normalization present or absent?
5. **Inspect weight structure**: Layer names, shapes, which layers pretrained?

### Phase 2: Architecture Design
1. **Choose base class**: BaseModel (simple) or DyRCNN (with biological features)
2. **Map layers to RConv2d**: Single-stage or two-stage? Where is recurrence?
3. **Define layer_operations**: What sequence replicates original?
4. **Plan temporal parameters**: What goes to RConv2d vs model-level?
5. **Identify special patterns**: Progressive activation? Custom initialization needed?

### Phase 3: Implementation
1. **Create model class** inheriting from base
2. **Implement `_define_architecture()`**: Define layers, operations, classifier
3. **Implement `reset()`**: Clear hidden states between forward passes
4. **Add pretrained loading** (if applicable): download, translate names, load
5. **Add custom operations** (if needed): Follow Lesson 1 guidelines

### Phase 4: Testing & Validation
1. **Test model creation**: Instantiate, check attributes exist
2. **Test forward pass**: Random input, verify output shape
3. **Create comparison test** (see Section below): Load original, compare outputs
4. **Validate weights**: Ensure all parameters match
5. **Validate layer-by-layer**: Hook and compare intermediate activations
6. **Validate timestep-by-timestep**: Check when divergence starts (if any)

### Phase 5: Debugging (If Needed)
1. **Apply validation hierarchy**: Don't jump to full model comparison
2. **Check temporal parameters**: Verify delay calculations
3. **Trace operation sequences**: Document both, compare explicitly
4. **Review relevant lessons**: Consult framework-specific sections below
5. **Iterate**: Fix one issue at a time, re-validate

### Time Investment Estimate
- **Investigation**: 2-4 hours (thorough understanding pays off)
- **Design**: 1-2 hours (planning prevents rewrites)
- **Implementation**: 2-6 hours (depends on complexity)
- **Testing**: 2-4 hours (incremental validation catches issues early)
- **Debugging**: 0-8 hours (good investigation minimizes this)

**Total**: 7-24 hours for typical model

### Critical Success Factors
✓ **Never skip investigation** - most debugging time comes from incomplete understanding
✓ **Use existing systems first** - check configuration/parameters before building new code
✓ **Validate incrementally** - catch issues at layer level, not after full integration
✓ **Document as you go** - future you (and others) will thank present you

---

## Core Principle: Investigation Before Implementation

**The Wrong Approach**:
1. See a requirement → immediately design a solution
2. Encounter unfamiliar code → guess at its behavior
3. Find complexity → build custom infrastructure
4. See a problem → assume existing code is wrong

**The Right Approach**:
1. **Investigate thoroughly**: Trace existing code, search for patterns, read source
2. **Understand constraints**: What exists? What can be reused? What must not change?
3. **Apply solution hierarchy**: Config → Parameter → Extension → New code
4. **Validate assumptions**: Never claim something is wrong without proof

---

## Lesson 1: Custom Operations and Execution Flow

### The Challenge
Some models require operations that must run even when `x=None` (e.g., initializing hidden states for layers that haven't received input yet).

### Framework Detail: temporal.py Execution Paths

The `_forward()` method in `temporal.py` processes operations with different execution conditions:

```python
# Path 1: Layer-specific modules (line 265-268)
elif hasattr(self, module_name):  # e.g., "operation_layername"
    module = getattr(self, module_name)
    if x is not None or "feedback" in operation or "skip" in operation:
        x = module(x)  # ← Only executes if x is not None!
```

**Implication**: If your operation needs to handle `None` inputs, **do not** name it using the `{operation}_{layer}` pattern.

```python
# Path 2: Fallback for non-standard names (line 270-272)
elif hasattr(self, operation) and x is not None:
    module = getattr(self, operation)
    x = module(x)  # ← Still checks x is not None
```

**Solution**: Use operation names that don't match standard patterns AND leverage the framework's else clause by ensuring the operation is defined per-layer with an unconventional base name.

### Guiding Recipe

**When you need operations that handle None inputs**:

1. **Choose a non-standard operation name** that won't collide with layer-specific naming
   - Avoid: `"addzeros"` (becomes `addzeros_V1`, matches Path 1 pattern)
   - Use: `"skipzeros"` or similar unique names

2. **Define per-layer modules** with this base name
   - `self.skipzeros_V1 = custom_function()`
   - `self.skipzeros_V2 = custom_function()`

3. **Place in correct position** within `layer_operations`
   - Usually AFTER `"delay"` to detect when feedforward signals are unavailable
   - The delay operation's output indicates signal availability

4. **Return None to preserve signal absence**
   - If input was None, return None
   - Initialization should be a side effect, not a data transformation

### Key Insight
Operation naming determines which execution path processes it. Understanding temporal.py's conditional logic is essential for custom operations that need special execution conditions.

---

## Lesson 2: Operation Order and Temporal Semantics

### The Challenge
Operations must execute in a specific order to correctly replicate temporal dynamics, especially regarding when hidden states are set vs. retrieved.

### Framework Detail: Delay Operation Mechanics

The `"delay"` operation (line 249-254 in temporal.py) does two things:

1. **Stores current state**: `layer.set_hidden_state(x)`
2. **Retrieves delayed state**: `x = layer.get_hidden_state(delay_feedforward + 1)`

At early timesteps, retrieval may return `None` if the buffer doesn't have enough history yet.

### Guiding Recipe

**When implementing models with progressive layer activation**:

1. **Understand the timing**:
   - At t=0, first layer processes input
   - At t=0, higher layers have no input yet (delay returns None)
   - Need to detect this None and initialize states

2. **Place initialization checks AFTER delay**:
   ```
   "layer"   → process (produces output or None)
   "norm"    → normalize
   "nonlin"  → activate
   "record"  → store for analysis
   "delay"   → set_hidden_state(x), then get_hidden_state(delay+1)
              → returns None if buffer insufficient
   "custom"  → detect None, initialize NEXT layer's states
   ```

3. **Initialize target layer, not current layer**:
   - Current layer's None output means next layer needs initialization
   - Don't initialize the layer producing None; initialize its downstream consumer

4. **Test temporal progression**:
   - Trace through t=0, t=1, t=2 manually
   - Verify which layers receive signals at each timestep
   - Confirm initialization happens at correct times

### Key Insight
The delay operation's dual nature (set current, get previous) creates a temporal window where signals are unavailable. Custom operations must be positioned to detect and respond to this unavailability.

---

## Lesson 3: Hidden State Initialization Semantics

### The Challenge
Original models may use different sentinel values for uninitialized states (scalar 0, None, zero tensors), and these differences affect integration operations.

### Framework Detail: Recurrence Integration

When RConv2d processes recurrence (in `forward_recurrence`, recurrence.py line 849-876):

```python
h = self.get_hidden_state(self.delay_recurrence)

if h is None:
    return x  # ← No recurrence applied, feedforward only
else:
    h = self.recurrence(h)
    x = self.integrate_signal(x, h)  # ← Additive by default
```

**Implication**: `None` vs zero tensor vs scalar 0 produce different behaviors:
- **None**: Recurrence completely skipped
- **Zero tensor**: Processed through recurrence operation (may have bias terms)
- **Scalar 0**: Only works with specific patching (not standard)

### Guiding Recipe

**When original model uses specific initialization values**:

1. **Identify original's initialization pattern**:
   - Look for `if state is None: state = 0` (scalar)
   - Look for `if state is None: state = torch.zeros(...)` (tensor)
   - Look for direct initialization: `state = initial_value`

2. **Choose appropriate DynVision solution**:

   **Case A: Original uses None semantics**
   - No action needed, DynVision default matches

   **Case B: Original uses zero tensors**
   - Implement custom initialization operation (Lesson 1)
   - Ensure zero tensors are stored in hidden state buffers
   - Verify recurrence type produces identity on zeros

   **Case C: Original uses scalar 0**
   - Generally avoid; requires extensive patching
   - Better: map to zero tensor initialization (Case B)

3. **Verify mathematical equivalence**:
   - Test: `recurrence(zeros) + feedforward` equals `feedforward`
   - For identity recurrence: use `fixed_self_weight=1.0`, `recurrence_bias=False`
   - This ensures `zeros * 1.0 + 0 = zeros`, making integration transparent

### Key Insight
Zero tensor initialization with identity recurrence replicates scalar 0 behavior mathematically. Both produce additive identity (x + 0 = x), but zero tensors integrate cleanly with DynVision's recurrence framework.

---

## Lesson 4: Parameter Scope and Responsibility Separation

### The Challenge
Knowing which temporal parameters to pass to RConv2d versus which are handled by the model-level temporal framework.

### Framework Detail: Dual Delay System

DynVision separates temporal dynamics into two concerns:

1. **Within-layer dynamics** (RConv2d responsibility):
   - Recurrent connections (this layer's past state)
   - Parameter: `t_recurrence` → calculates `delay_recurrence`
   - Method: `get_hidden_state(self.delay_recurrence)` within RConv2d

2. **Between-layer dynamics** (delay operation responsibility):
   - Feedforward delays (previous layer's output)
   - Parameter: `t_feedforward` → calculates `delay_feedforward` (model-level)
   - Method: `get_hidden_state(self.delay_feedforward + 1)` in temporal.py

### Guiding Recipe

**When configuring RConv2d layers**:

1. **Pass to RConv2d**:
   - `dt`: Integration timestep (needed for delay calculation)
   - `t_recurrence`: Recurrent delay within this layer
   - `history_length`: Must accommodate BOTH delays (see below)
   - `dim_y`, `dim_x`: Spatial dimensions for buffer allocation

2. **Do NOT pass to RConv2d**:
   - `t_feedforward`: Handled by delay operation in temporal.py
   - `t_feedback`, `t_skip`: Model-level concerns

3. **Calculate history_length carefully**:
   ```python
   # Must accommodate the LARGER of the two delays
   history_length = int(max(t_recurrence, t_feedforward) / dt) + 1
   ```
   Even though RConv2d doesn't use `t_feedforward` internally, its buffer must be large enough for both delay operations to retrieve states.

4. **Verify delay calculations**:
   - `delay_recurrence = int(t_recurrence / dt)` (inside RConv2d)
   - `delay_feedforward = int(t_feedforward / dt)` (model-level)
   - Check these match original model's temporal structure

### Key Insight
Respect the separation of concerns: RConv2d handles recurrence within a layer, the delay operation handles feedforward between layers. History length must accommodate both, even though each component only uses one delay value.

---

## Lesson 5: Solution Hierarchy - Start Simple

### The Challenge
When encountering requirements (like normalization overrides), there's temptation to build comprehensive new infrastructure.

### Framework Detail: Configuration Cascading

DynVision uses a hierarchical configuration system:
1. YAML config files (lowest priority, most flexible)
2. Pydantic parameter validation (handles type checking, defaults)
3. Command-line overrides (highest priority, runtime flexibility)

### Guiding Recipe

**When facing a new requirement, apply this hierarchy**:

1. **Configuration-only solution** (Try first):
   - Can changing a config value solve this?
   - Example: `snakemake --config normalize=null`
   - Cost: ~0 lines of code

2. **Parameter modification** (If config insufficient):
   - Can existing parameter validators handle new values?
   - Example: Update Pydantic validator to accept `None`
   - Cost: ~3-5 lines

3. **Extend existing code** (If new logic needed):
   - Can current functions/rules be enhanced?
   - Example: Add conditional check in existing Snakemake rule
   - Cost: ~5-10 lines

4. **New focused utility** (If isolated functionality needed):
   - Create minimal, single-purpose helper
   - Cost: 20-50 lines

5. **New abstraction** (Last resort):
   - Only if fundamentally new concept required
   - Cost: 100+ lines, maintenance burden

### The 10x Rule
If your proposed solution adds more than 10x the code of the simpler approach, reconsider. The simpler solution is usually hiding in existing systems.

### Key Insight
Most requirements can be solved at configuration or parameter levels. Exploring existing systems thoroughly before building prevents unnecessary complexity.

---

## Lesson 6: Tracing Before Claiming

### The Challenge
Architectural analysis can be subtle—operations split across different components (inside RConv2d vs layer_operations) can appear incorrect when they're actually equivalent.

### Framework Detail: Operation Decomposition

DynVision allows operations to be:
- **Inside RConv2d**: Via `mid_modules`, between conv stages
- **Outside RConv2d**: Via `layer_operations`, after RConv2d returns
- **Shared across layers**: Via `self.nonlin` accessed by all layers

This decomposition means identical operation sequences can be expressed multiple ways.

### Guiding Recipe

**When verifying architectural equivalence**:

1. **Document both implementations step-by-step**:
   ```
   Original:
   1. conv_input(x)    → [3, 64, 56, 56]
   2. norm_input(x)    → [3, 64, 56, 56]
   3. nonlin_input(x)  → [3, 64, 56, 56]
   4. x + state        → [3, 64, 56, 56] (recurrence point)
   5. conv1(x)         → [3, 64, 56, 56]
   6. norm1(x)         → [3, 64, 56, 56]
   7. nonlin1(x)       → [3, 64, 56, 56]

   DynVision:
   [RConv2d internal]
   1. self.conv(x)     → [3, 64, 56, 56]
   2. mid_modules(x)   → [3, 64, 56, 56] (norm_input)
   3. internal_nonlin  → [3, 64, 56, 56]
   4. recurrence(h)    → [3, 64, 56, 56] (x + recurrence point)
   5. self.conv2(x)    → [3, 64, 56, 56]
   [layer_operations]
   6. norm_V1(x)       → [3, 64, 56, 56] (norm1)
   7. nonlin(x)        → [3, 64, 56, 56] (nonlin1)

   Result: Sequences IDENTICAL
   ```

2. **Count operations explicitly**:
   - How many operations before recurrence?
   - How many after?
   - Where do shapes change?

3. **Never claim "wrong" without end-to-end trace**:
   - Complete both traces first
   - Compare operation-by-operation
   - Only then identify actual differences

4. **Test with identical inputs**:
   - Same random seed
   - Same preprocessing
   - Same device/dtype
   - Compare intermediate activations

### Key Insight
Intuition about "where things should be" is unreliable. Always trace both implementations completely before making architectural claims.

---

## Lesson 7: Framework Component Capabilities

### The Challenge
Building custom infrastructure for capabilities that already exist in framework components.

### Framework Detail: RConv2d Built-in Features

RConv2d (via ForwardRecurrenceBase) already provides:
- `get_hidden_state(delay)`: Retrieve state from N timesteps ago
- `set_hidden_state(h)`: Store state for future retrieval
- `reset()`: Clear all hidden states
- DataBuffer: Circular buffer for efficient state storage
- Automatic delay calculation: `delay_recurrence = int(t_recurrence / dt)`

### Guiding Recipe

**Before implementing helper functions**:

1. **Search existing components**:
   ```bash
   grep -r "def method_name" dynvision/model_components/
   grep -r "class.*Base" dynvision/base/
   ```

2. **Read parent class source**:
   - ForwardRecurrenceBase (state management)
   - TemporalBase (temporal dynamics)
   - BaseModel/DyRCNN (model coordination)

3. **Check if RConv2d with recurrence_type="none" suffices**:
   - Provides state management without recurrence
   - Can replace custom "stateful wrappers"

4. **Leverage auto_adapt in Skip/Feedback**:
   - Automatically learns shape transformations
   - No need for custom dimension matching

### Key Insight
Framework components often provide more functionality than initially apparent. Reading source code before implementing reveals existing capabilities.

---

## Lesson 8: Recurrence Type Configuration

### The Challenge
Understanding how recurrence types map to original implementation's connection patterns.

### Framework Detail: Recurrence Types

Available in DynVision (recurrence.py):
- `"none"`: No recurrence, feedforward only
- `"self"`: Point-wise scaling (same channels, 1x1 effective)
- `"full"`: Full convolution (3x3 typical, learnable)
- `"depthwise"`: Depthwise separable
- `"local"`: Locally connected (position-specific)

Key parameters:
- `fixed_self_weight`: For "self", makes weight non-trainable (e.g., 1.0 for identity)
- `recurrence_bias`: Whether to include bias in recurrence connection
- `recurrence_target`: Where to apply ("input", "middle", "output")

### Guiding Recipe

**When mapping original recurrence to DynVision**:

1. **Identify original's recurrent connection**:
   - What are input/output channels? (same → "self", different → consider "full")
   - What is kernel size? (1x1 → "self", 3x3 → "full")
   - Is weight learnable or fixed? (fixed → `fixed_self_weight`)
   - Is there bias? (no → `recurrence_bias=False`)

2. **For identity recurrence** (like CorNet-RT):
   - Use `recurrence_type="self"`
   - Set `fixed_self_weight=1.0` (non-trainable)
   - Set `recurrence_bias=False`
   - Result: `h_out = h_in * 1.0 + 0 = h_in` (mathematical identity)

3. **For learnable recurrence**:
   - Use `recurrence_type="full"` for spatial convolution
   - Use `recurrence_type="self"` for channel-wise scaling
   - Leave `fixed_self_weight=None` (default, makes it learnable)

4. **Verify at recurrence target**:
   - Count operations in original to find where state is added
   - Map to "input", "middle", or "output" accordingly

### Key Insight
Identity recurrence (`fixed_self_weight=1.0`, no bias) replicates additive recurrence patterns (x + state) when state is properly initialized. Zero tensors through identity recurrence remain zero (transparent).

---

## Lesson 9: Incremental Validation Methodology

### The Challenge
When outputs don't match despite all individual checks passing, systematic debugging is essential.

### Framework Detail: Multi-Level Validation

DynVision's modular design allows validation at different granularities:
- Weight level (parameters match)
- Layer level (intermediate activations match)
- Timestep level (temporal dynamics match)
- Output level (final predictions match)

### Guiding Recipe

**Apply validation hierarchy when debugging**:

1. **Weight Validation**:
   - Compare all parameter shapes
   - Compare all parameter values (within floating point tolerance)
   - Check translation mapping is correct
   - Verify pretrained weights loaded

2. **Single Layer Validation**:
   - Hook one layer in both implementations
   - Use identical input
   - Compare outputs AFTER same post-processing
   - Mind the hook capture point (before or after norm/nonlin)

3. **Single Timestep Validation**:
   - Compare t=0 only (no recurrence effects)
   - Should match exactly if weights match
   - Isolates recurrence vs feedforward issues

4. **Multi-Timestep Validation**:
   - Compare t=0, t=1, t=2 separately
   - Identify at which timestep divergence begins
   - Points to temporal parameter issues

5. **Full Model Validation**:
   - Only after above pass
   - Use identical random seeds, preprocessing, device
   - Compare final predictions

### Debug Checklist

When validation fails:
- [ ] Are weights identical? (check translation, check loading)
- [ ] Are operations in same order? (trace both implementations)
- [ ] Are temporal parameters correct? (check delay calculations)
- [ ] Are hooks capturing at same point? (pre vs post norm/nonlin)
- [ ] Are inputs truly identical? (seed, preprocessing, normalization)
- [ ] Is hidden state initialization consistent? (None vs zeros vs scalar)

### Key Insight
Never skip validation levels. Each level isolates different potential issues. Jumping to full model comparison without layer/timestep validation wastes time.

---

## Lesson 10: Building a Comprehensive Test Script

### The Challenge
A well-structured test script is essential for validating reimplementation correctness. It should test incrementally, provide clear diagnostics, and pinpoint exactly where divergence occurs.

### Test Script Structure

A complete test script should have four main components:

#### Component 1: Model Loading Infrastructure

**Purpose**: Consistently load both models in evaluation mode with pretrained weights

**Key considerations**:
- Load original model from reference implementation
- Load DynVision reimplementation with `init_with_pretrained=True`
- Set both to `eval()` mode (disables dropout, batch norm training mode)
- Use same device (CPU or GPU)
- Disable gradients (`torch.no_grad()`)

**What to include**:
- Clear separation between original and reimplemented model setup
- Parameter count verification (should be nearly identical)
- Configuration display (timesteps, delays, etc.)

#### Component 2: Weight Comparison

**Purpose**: Verify pretrained weights loaded correctly before testing outputs

**Comparison strategy**:
1. Get state dictionaries from both models
2. Create translation mapping (original names → DynVision names)
3. Compare only matching parameters (exclude classifier if different n_classes)
4. Check shape match first, then value match
5. Report mismatches clearly

**Tolerance guidelines**:
- Shape mismatches: Show both shapes
- Value mismatches: Show max absolute difference
- Success threshold: All parameters within 1e-6 (floating point precision)

**Critical insight**: If weights don't match, outputs won't match. Fix weight loading before proceeding.

#### Component 3: Input Preparation

**Purpose**: Create truly identical inputs for both models

**Critical requirements**:
- **Same random seed**: Set `torch.manual_seed()` and `np.random.seed()`
- **Same preprocessing**: Apply identical transforms (resize, crop, normalization)
- **Same data format**: Original may expect [B,C,H,W], reimplemented expects [B,T,C,H,W]
- **Same device**: Both inputs on same device

**Temporal dimension handling**:
- Original models often loop internally over same image
- Reimplemented models expect temporal dimension explicitly
- Solution: Repeat image along temporal axis: `img.unsqueeze(1).repeat(1, n_timesteps, 1, 1, 1)`

**Normalization considerations**:
- Check if original model expects normalized inputs or raw pixels
- Apply same normalization to both (or neither)
- Document this in test output for reproducibility

#### Component 4: Multi-Level Validation

**Level 1: Final Output Comparison**

**Purpose**: Quick sanity check - do final predictions match?

**Metrics to compute**:
- Mean absolute difference: `(out_orig - out_reimpl).abs().mean()`
- Max absolute difference: `(out_orig - out_reimpl).abs().max()`
- Top-k prediction agreement: Do both predict same class?
- Statistical measures: Compare mean and std of outputs

**Interpretation**:
- Perfect match: Mean diff < 1e-4, predictions identical
- Small numerical error: Mean diff < 1e-2, predictions identical (acceptable)
- Divergence: Mean diff > 1e-2 or different predictions (investigate)

**Level 2: Layer-by-Layer Comparison**

**Purpose**: Identify which layer first diverges

**Hook placement strategy**:
1. Register forward hooks on each layer in both models
2. Store activations in dictionaries keyed by layer name
3. Run both models with identical input
4. Compare stored activations layer by layer

**Critical hook considerations**:
- Hook captures at registration point (pre or post norm/nonlin)
- For fair comparison, apply same post-processing to both
- Original layers may be composite (multiple operations)
- DynVision separates operations (RConv2d, then norm, then nonlin)

**What to compare**:
- Activation shapes (should match)
- Activation statistics (mean, std)
- Absolute differences (mean, max)
- Relative differences (normalized by magnitude)

**Interpretation guidelines**:
- If Layer N matches but Layer N+1 diverges:
  - Check operation order between layers
  - Check delay parameters
  - Check if initialization needed
- If all layers diverge from start:
  - Check weight loading
  - Check input preprocessing
  - Check operation sequence

**Level 3: Temporal Progression Analysis**

**Purpose**: For recurrent models, identify when divergence begins

**Temporal validation strategy**:
1. Manually step through original model's time loop
2. Compare with reimplemented model's responses at each timestep
3. Track which layers are active at each timestep
4. Compare activations timestep-by-timestep

**What to track**:
- t=0: Which layers receive input? (Usually first layer only)
- t=1: Which new layers activate? (Progressive activation)
- t=2+: All layers active, compare states
- Hidden state contents at each timestep

**Diagnostic questions**:
- Does divergence start at t=0 or later? (Later → recurrence issue)
- Does divergence affect all layers or specific ones? (Specific → that layer's config)
- Does divergence accumulate over time? (Yes → temporal parameter mismatch)

### Test Script Best Practices

#### Structure and Organization

**Use a class-based approach**:
- `ModelLoader`: Handles loading both models
- `ModelComparator`: Performs comparisons
- `ComparisonMetrics`: Dataclass for storing results
- `main()`: Orchestrates the test sequence

**Separation of concerns**:
- Loading logic separate from comparison logic
- Each validation level in its own method
- Clear, descriptive method names
- Minimal coupling between components

#### Output and Reporting

**Progressive disclosure**:
- Show high-level summary first (weights match? outputs match?)
- Provide detailed breakdowns on request or on failure
- Use visual separators (===== sections)
- Color/symbols for quick scanning (✓ ✗)

**Informative error messages**:
- Not just "failed" but "Layer V2 diverged: mean_diff=0.007"
- Include expected vs actual values
- Suggest what to check next
- Reference relevant lessons in this document

#### Reproducibility

**Document test conditions**:
- Random seed used
- Device used (CPU/GPU)
- Model configurations
- Input preprocessing applied

**Save artifacts** (optional but helpful):
- Comparison results to JSON
- Activation differences to file
- Model outputs for offline analysis

### Validation Workflow

**Step 1: Run Weight Comparison**
- If fails: Fix weight loading, translation mapping
- If passes: Proceed to output comparison

**Step 2: Run Output Comparison**
- If matches: Success! Done.
- If diverges slightly: Acceptable if predictions agree
- If diverges significantly: Proceed to layer comparison

**Step 3: Run Layer-by-Layer Comparison**
- Identify first diverging layer
- Check that layer's configuration
- Check operations between previous and diverging layer
- If still unclear: Proceed to temporal analysis

**Step 4: Run Temporal Progression Analysis**
- Trace timestep-by-timestep
- Identify when divergence starts
- Check temporal parameters
- Check hidden state initialization

**Step 5: Consult Relevant Lessons**
- Diverges at t=0: Check Lesson 6 (tracing), Lesson 3 (initialization)
- Diverges at t>0: Check Lesson 2 (operation order), Lesson 4 (parameters)
- Specific layer diverges: Check Lesson 8 (recurrence config)
- All layers diverge: Check Lesson 5 (solution hierarchy - is preprocessing correct?)

### Test Script Template Structure

A complete test script should follow this organization:

```
1. Imports and Setup
   - Import both models
   - Import comparison utilities
   - Set up logging/printing

2. Data Classes
   - ComparisonMetrics (stores comparison results)
   - Configuration classes (if needed)

3. ModelLoader Class
   - load_original() -> original_model, state_dict
   - load_reimplemented() -> reimplemented_model

4. ModelComparator Class
   - __init__(original, reimplemented, device)
   - compare_weights(orig_state_dict) -> bool
   - create_inputs(batch_size, seed) -> dict of inputs
   - compare_outputs(inputs) -> ComparisonMetrics
   - compare_layer_activations(inputs) -> dict of metrics per layer
   - compare_temporal_progression(inputs) -> temporal analysis

5. Main Function
   - Load both models
   - Run weight comparison
   - Create test inputs
   - Run output comparison
   - Run layer comparison (if needed)
   - Run temporal analysis (if needed)
   - Print summary

6. Entry Point
   - if __name__ == "__main__": main()
```

### Common Test Script Pitfalls

**Pitfall 1: Hooks Capture at Wrong Point**
- Symptom: Layer activations appear different despite correct implementation
- Solution: Ensure hooks capture after same post-processing in both models
- Check: Are you comparing pre-norm vs post-norm?

**Pitfall 2: Different Random Seeds**
- Symptom: Outputs differ on different runs
- Solution: Always set seed before creating inputs
- Check: Set both `torch.manual_seed()` and `np.random.seed()`

**Pitfall 3: Temporal Dimension Mismatch**
- Symptom: Shape errors or unexpected behavior
- Solution: Original [B,C,H,W], reimplemented [B,T,C,H,W]
- Check: Repeat image along time dimension for reimplemented

**Pitfall 4: Comparing Wrong Timesteps**
- Symptom: Outputs differ but architecturally correct
- Solution: Original's final output = reimplemented's final timestep
- Check: Compare `out_orig` with `out_reimpl[:, -1, ...]`

**Pitfall 5: Insufficient Tolerance**
- Symptom: Test reports failure despite functionally correct
- Solution: Use appropriate tolerance (1e-4 to 1e-6 for float32)
- Check: Are predictions identical even if values differ slightly?

### Key Insight
A well-structured test script is an investment that pays off immediately. It should:
- Run quickly (use small batch size, few timesteps)
- Report clearly (exactly where and by how much things differ)
- Guide debugging (suggest what to check based on failure pattern)
- Be maintainable (clean structure, documented assumptions)

Good testing reveals issues early and precisely, bad testing wastes time with vague failures.

---

## Summary: The Model Integration Mindset

### Investigation Over Implementation
1. Trace original code completely before claiming understanding
2. Search existing systems before building new infrastructure
3. Validate assumptions with data, not intuition

### Framework-Aware Development
1. Understand temporal.py execution paths for custom operations
2. Respect parameter scope (RConv2d vs model-level)
3. Leverage existing component capabilities

### Systematic Validation
1. Apply validation hierarchy (weights → layers → timesteps → output)
2. Test incrementally, not in one big bang
3. Document operation sequences explicitly

### Questions to Ask Before Implementing

- **Have I traced both implementations end-to-end?**
- **Does existing infrastructure handle this requirement?**
- **Am I using the simplest solution (config → param → extension → new)?**
- **Do I understand which component is responsible for this behavior?**
- **Have I validated at each level (weights, layers, timesteps)?**

---

## Integration with Main Guide

This lessons document should be read **after** working through the main [Model Integration Guide](model-integration.md) and encountering your first challenging implementation. The main guide provides the procedural framework; this document provides the hard-won wisdom about framework-specific details and pitfalls.

**Recommended workflow**:
1. Read main integration guide for overall process
2. Start your implementation following the phases
3. When stuck, consult specific lessons here
4. Return to main guide for next phase

---

*"Before building new, understand what exists. Before coding, trace the original. Before claiming equivalence, validate with data. Before adding complexity, exhaust simplicity."*
