# Temporal Data Presentation & Loss Reaction Time Plan

## Progress Log
- **2025-11-20** – Document drafted; captured current vs. target behavior, relocation plan, and pattern-aware masking strategy.
- **2025-11-20** – Began implementation: moved `loss_reaction_time` + `_init_loss` ownership into `TemporalBase`, removed Lightning-specific masking, and added per-chunk reaction masking/warnings wired through `_expand_timesteps`.
- **2025-11-20** – Updated presentation shuffling so the original pattern entries (pre-resampling) are permuted before expansion to `n_timesteps`, ensuring each shuffled stimulus/null block retains its intended duration.

## Context
- Temporal dynamics live in `dynvision/base/temporal.py` (`TemporalBase`), while PyTorch Lightning glue is in `dynvision/base/lightning.py` (`LightningBase`).
- Presentation patterns (`data_presentation_pattern`, optional shuffling, null-input masking) are handled in `TemporalBase._expand_timesteps` and `_cache/_get_presentation_pattern`.
- Loss criteria initialization (`_init_loss`) and `loss_reaction_time` handling currently sit in `LightningBase`, where the reaction window is applied once at the start of every sequence by zeroing a fixed prefix of labels. This ignores per-stimulus presentation chunks and temporal shuffling.
- Goal: move ownership of `loss_reaction_time` into `TemporalBase` and make the masking pattern-aware so every new stimulus chunk after a null input honors the reaction window. Warn when the reaction window exceeds a chunk’s duration.

## Current Behaviour (Nov 20, 2025)
1. `LightningBase.__init__` stores `loss_reaction_time` (default 4 ms) and builds losses via `_init_loss()`.
2. `_init_loss()` computes `ignore_initial_n_labels = self.n_residual_timesteps + int(self.loss_reaction_time / self.dt)` and masks that many leading timesteps in `compute_loss()` by setting their labels to `non_label_index`.
3. Presentation pattern masking happens earlier in `TemporalBase._expand_timesteps`, but `loss_reaction_time` is unaware of the pattern or shuffled order.
4. Result: only the first reaction window in the entire clip is ignored, regardless of how many distinct stimulus presentations occur later.

## Target Behaviour
- `TemporalBase` owns `loss_reaction_time` (alias `lossrt`) and performs all label masking tied to temporal dynamics.
- For each batch, the effective presentation mask (after shuffling/resampling) is used to derive contiguous "presentation chunks" (runs of 1s delimited by 0s).
- Convert `loss_reaction_time` into time steps: `reaction_steps = ceil(loss_reaction_time / dt)`.
- For every chunk, zero-out the first `reaction_steps` positions (or the entire chunk if shorter) by reusing the existing label-masking mechanism (set to `non_label_index`).
- Emit a `logger.warning` whenever `reaction_steps * dt` exceeds the chunk duration so users know the loss never sees that chunk.
- Because patterns can reshuffle per batch, recompute the masking every `_expand_timesteps` call; no long-lived cache for reaction windows.

## Implementation Plan

### Status (2025-11-20)
- ✅ Loss configuration + `_init_loss` now live in `TemporalBase`, called from its new `setup` hook before delegating to Lightning/Mixins.
- ✅ `LightningBase` no longer owns `loss_reaction_time`, `_init_loss`, or the fixed-prefix masking.
- ✅ `_expand_timesteps` now computes a per-batch reaction mask from the (possibly shuffled) presentation pattern and voids labels for the first `ceil(loss_reaction_time / dt)` steps of every chunk, warning when the chunk is shorter than the reaction window.

### 1. Move configuration + loss init to `TemporalBase`
- Extend `@alias_kwargs` in `TemporalBase` with `lossrt="loss_reaction_time"`.
- Add `loss_reaction_time: float = 0.0` parameter, store via `float()` (mirroring existing attributes).
- Port `_init_loss()` and related imports from `LightningBase` into `TemporalBase`. Keep identical logic (criterion weights, ignore_index auto-fill, hook registration) but drop the old `ignore_initial_n_labels` math—it becomes pattern-aware later.
- Provide a `setup()` override inside `TemporalBase` that calls `self._init_loss()` then `super().setup(stage)` so downstream classes (e.g., LightningBase) still run their hooks.
- Remove redundant `_init_loss()` and `setup()` overrides from `LightningBase`; its `compute_loss()` should now assume labels arrive pre-masked.

### 2. Pattern-aware reaction masking
- Inside `_expand_timesteps`, after retrieving `presentation_pattern` (already a bool tensor length `self.n_timesteps`), build a helper `_compute_reaction_mask(pattern: torch.Tensor) -> torch.Tensor` that returns a bool mask with the same length where `True` indicates timesteps to ignore for loss.
- Helper algorithm:
  1. If `self.loss_reaction_time <= 0`, return `pattern.new_zeros(pattern.shape)`.
  2. Compute `reaction_steps = max(1, math.ceil(self.loss_reaction_time / self.dt))`.
  3. Detect chunk boundaries: `starts = pattern & torch.cat([pattern[:1], pattern[1:] & ~pattern[:-1]])` or via `torch.diff` logic to find rising edges. Iterate through chunks via index arithmetic (vectorized if possible; loop acceptable because `n_timesteps` is small).
  4. For each chunk `[start, end)`, set mask slice `start : min(start + reaction_steps, end)` to `True`.
  5. If `reaction_steps > (end - start)`, log a warning once per chunk: include chunk length (converted to ms using `dt`) and `loss_reaction_time`.
- Back in `_expand_timesteps`, after cloning expanded tensors (already required for presentation masking), apply the reaction mask exactly like the existing `zero_mask`: set `label_indices[:, reaction_mask] = self.non_label_index`. Inputs shouldn’t be zeroed—stimulus is visible—but labels must be ignored so the loss isn’t evaluated.
- Combine masks safely: use `mask_to_apply = zero_mask | reaction_mask` so both null inputs and reaction windows receive the void label.

### 3. Plumbing into `compute_loss`
- With labels already masked per chunk, `LightningBase.compute_loss()` no longer needs to clone-and-mask the leading window. Remove references to `ignore_initial_n_labels` there.
- Retain the validation that logs when all labels are invalid—it now catches cases where reaction time wipes out entire sequences.

### 4. Diagnostics & Documentation
- Add warning helper inside `_compute_reaction_mask` so repeated warnings don’t spam logs: keep `self._warned_chunks` set or throttle per forward pass.
- Update docstrings/comments near `_expand_timesteps` to describe the new behavior.
- Extend this planning doc as implementation progresses; later, summarize in `docs/development/guides/claude-guide.md` and user-facing docs once code lands.

## Testing & Validation Ideas
- Unit-ish check: craft a tiny `TemporalBase` subclass with `n_timesteps=8`, `dt=5`, pattern `10001111`, `loss_reaction_time=6`. After `_expand_timesteps`, verify that labels for timesteps `[0,1]` and `[4]` (start of second chunk) are set to `non_label_index` and that warnings trigger because chunk durations (10 ms and 20 ms) are shorter than reaction time.
- Ensure per-batch shuffling still works: run two `_expand_timesteps` calls with `shuffle_presentation_pattern=True` and confirm the reaction mask realigns with each shuffled pattern.
- Regression: confirm Lightning training still logs loss values and `criterion` hooks remain intact.

## Open Questions / Next steps
1. **Performance:** If `n_timesteps` is large, do we need vectorized chunk detection? (Probably fine to start with simple loops.)
2. **User control:** Should we expose a flag to disable chunk-wise masking even when `loss_reaction_time>0`? (Not currently requested; document default.)
3. **Docs:** Once implemented, integrate this plan into the temporal modeling guide and mention the warning semantics.

---

## Loss Calculation Review (2025-11-21)

### Summary
Reviewed the complete loss calculation pipeline including normalization, compression, and combination to ensure correct handling of valid vs. invalid timesteps across multiple loss types.

### Loss Calculation Flow
1. **LightningBase.model_step** (dynvision/base/lightning.py:85-167) → calls **compute_loss** (line 121-124)
2. **LightningBase.compute_loss** (dynvision/base/lightning.py:261-304) → flattens outputs/labels, iterates through criteria
3. Each criterion's **forward** method → calls **compute_loss** then **apply_reduction** from BaseLoss

### CrossEntropyLoss: ✅ Correct Normalization
**Location:** `dynvision/losses/cross_entropy_loss.py`, `dynvision/losses/base_loss.py`

**Behavior:**
- `CrossEntropyLoss.compute_loss` (lines 28-57):
  - Computes element-wise cross-entropy with `reduction="none"` (line 41-47)
  - Creates `valid_mask` excluding `ignore_index` entries (line 50)
  - Zeros out invalid entries by multiplying loss by mask (line 54)
  - Returns element-wise losses (no reduction)

- `BaseLoss.forward` (lines 98-125):
  - Infers `num_valid_timesteps` from targets if criterion has `ignore_index` (lines 117-122)
  - Counts valid timesteps: `mask = (targets != ignore_index)`, `num_valid = mask.sum()` (lines 119-120)
  - Passes `num_valid_timesteps` to `apply_reduction` (line 125)

- `BaseLoss.apply_reduction` (lines 72-85):
  - When `reduction="mean"` and `num_valid_timesteps` provided: `loss.sum() / num_valid_timesteps` (lines 76-81)
  - This correctly normalizes **only by valid timesteps**, excluding masked-out timesteps

**Result:** CrossEntropyLoss correctly normalizes by the count of valid (non-masked) timesteps only.

### EnergyLoss: ⚠️ Issues Identified
**Location:** `dynvision/losses/energy_loss.py`

**Current Behavior:**
- `EnergyLoss.forward` (lines 58-68):
  - Directly calls `compute_loss()` then `apply_reduction(loss)` (lines 66-68)
  - **Does NOT pass `num_valid_timesteps`** to `apply_reduction`
  - BaseLoss.apply_reduction defaults to simple `loss.mean()` over batch dimension

- `EnergyLoss.compute_loss` (lines 70-109):
  - Hooks registered on monitored modules (Conv2d, Linear, etc.) capture activations during forward pass (lines 21-56)
  - `_accumulate_energy` computes per-batch energy norm: `torch.norm(activation, p=self.p, dim=tuple(range(1, activation.ndim)))` (lines 45-46)
  - Returns shape `[batch_size]` averaged over monitored modules (line 105)

**Issues:**
1. **Overwrites instead of accumulating:** Line 54 sets `self.batch_energy[module_name] = batch_energy`, which **overwrites** the previous value on each hook call. Since the temporal model calls layers once per timestep (temporal.py:399-407), hooks fire `n_timesteps` times but only the **last timestep's energy is captured**.

2. **No timestep normalization:** Even if accumulation worked, there's no division by `n_timesteps` or distinction between valid/invalid timesteps.

3. **No temporal awareness:** EnergyLoss doesn't account for presentation patterns, reaction time masking, or null inputs. It treats all timesteps equally.

**Expected Behavior:**
- EnergyLoss should accumulate energy across all timesteps during the forward pass
- When `reduction="mean"`, it should normalize by the total number of timesteps (not just batch size)
- Whether to exclude invalid timesteps is a design decision (typically energy should count all computational activity)

### Loss Combination: ✅ Correct
**Location:** `dynvision/base/lightning.py:261-304`

**Behavior:**
- Line 270-271: Flattens outputs `[batch, timesteps, classes]` → `[batch*timesteps, classes]` and labels
- Line 280: Initializes loss accumulator
- Lines 282-296: Iterates through each criterion:
  - Extracts weight from tuple `(criterion_fn, weight)` (line 283-285)
  - Computes weighted loss: `loss_value = weight * criterion_fn(outputs, label_index)` (line 288)
  - Logs individual loss values (lines 291-296)
- Line 298: Sums all weighted losses: `loss = loss_values.sum()`

**Result:** Loss combination correctly weights and sums individual criterion losses.

### Recommendations
1. ✅ **EnergyLoss accumulation:** Modify `_accumulate_energy` to accumulate energy across timesteps rather than overwrite
2. ✅ **EnergyLoss normalization:** Add proper normalization by total timesteps when reduction="mean"
3. ✅ **Documentation:** Document that EnergyLoss computes total energy across all timesteps (including invalid ones) vs. CrossEntropyLoss which only considers valid timesteps
4. ✅ **Testing:** Add unit tests to verify:
   - CrossEntropyLoss normalization with masked timesteps
   - EnergyLoss accumulation across timesteps
   - Weighted loss combination

---

## Implementation Completed (2025-11-21)

### Changes Made

#### 1. EnergyLoss Accumulation (dynvision/losses/energy_loss.py:52-80)
**Before:** Line 62 overwrote energy with `self.batch_energy[module_name] = batch_energy`

**After:** Lines 71-77 accumulate energy across timesteps:
```python
if module_name not in self.batch_energy:
    self.batch_energy[module_name] = batch_energy
    self._hook_call_count[module_name] = 1
else:
    self.batch_energy[module_name] = self.batch_energy[module_name] + batch_energy
    self._hook_call_count[module_name] += 1
```

**Rationale:** Hooks fire once per monitored layer per timestep during `TemporalBase.forward()`. By accumulating rather than overwriting, we capture the total energy across all timesteps.

#### 2. EnergyLoss Timestep Normalization (dynvision/losses/energy_loss.py:94-153)
**Added:** Lines 112-120 infer `n_timesteps` from hook call counts:
```python
n_timesteps = 1
if self._hook_call_count:
    call_counts = list(self._hook_call_count.values())
    n_timesteps = max(call_counts) if call_counts else 1
```

**Modified:** Lines 140-143 normalize by both modules and timesteps:
```python
loss = total_energy / (module_count * n_timesteps)
```

**Rationale:** The accumulated energy represents the sum over all timesteps. Dividing by `n_timesteps` gives the average energy per timestep, which when combined with the existing normalization by `n_units` (spatial dimensions) and `module_count`, yields the **average activity per unit per timestep per module**.

#### 3. Enhanced Documentation (dynvision/losses/energy_loss.py:8-20)
Added comprehensive docstring explaining:
- Energy accumulation across all timesteps
- Normalization behavior when `reduction='mean'`
- Contrast with CrossEntropyLoss (all timesteps vs. valid timesteps only)
- Hook firing pattern and timing

#### 4. Unit Tests (tests/losses/test_loss_normalization.py)
Created comprehensive test suite with 6 tests:

**TestCrossEntropyLossNormalization:**
- `test_normalization_with_valid_timesteps_only`: Verifies normalization by valid timesteps only
- `test_all_timesteps_masked`: Verifies behavior when all labels are masked

**TestEnergyLossAccumulation:**
- `test_accumulation_across_timesteps`: Verifies energy accumulates correctly over multiple timesteps
- `test_hook_call_count_tracking`: Verifies hook call counts track timesteps correctly
- `test_multiple_modules_averaging`: Verifies averaging across multiple monitored modules

**TestLossCombination:**
- `test_weighted_loss_combination`: Verifies weighted sum of multiple losses

**All tests pass:** ✅ 6/6 passed

### Verification of Normalization

**For p=1 (L1 norm, default):**
1. Hook computes: `||activation||_1` per timestep → shape `[batch_size]`
2. Accumulated over timesteps: `Σ_t ||activation_t||_1`
3. Normalized by `n_units`: `Σ_t ||activation_t||_1 / n_units` = `Σ_t (mean_units |a_t|)`
4. Normalized by `n_timesteps`: `mean_t (mean_units |a_t|)` = **average absolute activity per unit per timestep**
5. Averaged over modules: **average absolute activity per unit per timestep per module**
6. Averaged over batch: **final scalar loss**

**Result:** EnergyLoss correctly measures average absolute activity per unit, per timestep, per module, per sample.

### Summary
The loss calculation pipeline now correctly handles:
- ✅ **CrossEntropyLoss:** Normalizes by valid (non-masked) timesteps only
- ✅ **EnergyLoss:** Accumulates and normalizes across all timesteps
- ✅ **Loss Combination:** Weights and sums losses correctly
- ✅ **Documentation:** Clear explanation of normalization semantics
- ✅ **Testing:** Comprehensive unit tests verify all behaviors

---

## GPU Accelerator Auto-Detection Fix (2025-11-21)

### Issue
Training ran on CPU despite GPU being available. PyTorch Lightning showed:
```
GPU available: True (cuda), used: False
PossibleUserWarning: GPU available but not used. Set `accelerator` and `devices`
```

All model parameters were on CPU device instead of cuda:0.

### Root Cause
**Location:** `dynvision/params/trainer_params.py:644-649`

When using single-device training (non-distributed, `world_size=1`) with default `accelerator="auto"`:
1. The code only set `accelerator` if it wasn't "auto": `if self.accelerator != "auto"`
2. With the default config having `accelerator: auto`, the accelerator was never explicitly set
3. PyTorch Lightning then defaulted to CPU when accelerator wasn't specified

### Fix
**Modified:** `dynvision/params/trainer_params.py:644-657`

Changed the single-device training logic to:
```python
else:
    # For single device training
    if self.accelerator == "auto":
        # Auto-detect: use GPU if available, otherwise let Lightning default to CPU
        if self._detect_available_gpu_count() > 0:
            trainer_kwargs["accelerator"] = "gpu"
            # Use configured devices or default to 1 for single-device training
            trainer_kwargs["devices"] = self.devices if self.devices is not None else 1
        # If no GPU available, let Lightning default to CPU (don't set accelerator)
    else:
        # User explicitly specified accelerator
        trainer_kwargs["accelerator"] = self.accelerator
        if self.devices is not None:
            trainer_kwargs["devices"] = self.devices
```

**Behavior:**
- When `accelerator="auto"` in single-device mode:
  - Detects available GPUs using existing `_detect_available_gpu_count()` method
  - If GPUs available: sets `accelerator="gpu"` and `devices` (respects user config or defaults to 1)
  - If no GPUs: lets Lightning default to CPU (no accelerator set)
- When accelerator explicitly specified: uses user's configuration as before

**Result:** Single-device training now automatically uses GPU when available, matching user expectations for the "auto" setting.

_This document will be updated as implementation proceeds._
