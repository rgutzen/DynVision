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

## Loss Calculation Implementation (2025-11-21)

### Summary
Reviewed and fixed the complete loss calculation pipeline to ensure correct handling of valid vs. invalid timesteps across multiple loss types.

**Key Findings:**
- ✅ **CrossEntropyLoss:** Normalizes by valid (non-masked) timesteps only
- ⚠️ **EnergyLoss:** Was overwriting instead of accumulating energy across timesteps
- ✅ **Loss Combination:** Correctly weights and sums individual losses

### Fixes Applied

1. **EnergyLoss Accumulation** - Modified `_accumulate_energy()` to accumulate energy across timesteps rather than overwrite
2. **EnergyLoss Timestep Normalization** - Infer `n_timesteps` from hook call counts and normalize by total timesteps
3. **Device Handling** - Added automatic device alignment for GPU/CPU transfers
4. **GPU-CPU Sync Elimination** - Rewrote `_compute_reaction_mask()` using fully vectorized operations
5. **Unit Tests** - Created `tests/losses/test_loss_normalization.py` with 6 comprehensive tests

**Result:** EnergyLoss now correctly measures average absolute activity per unit, per timestep, per module, per sample.

**Detailed documentation:** See [Loss Functions Reference](../reference/losses.md)

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
