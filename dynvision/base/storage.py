"""
Enhanced data buffer for DynVision PyTorch Lightning workflows.

Provides efficient storage for neural network responses and records with
flexible strategy alignment, unlimited storage option, and improved memory management.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import threading
import gc
from pytorch_lightning import LightningModule

import torch
import pandas as pd
import numpy as np
import time

logger = logging.getLogger(__name__)


class SamplingStrategy:
    """Base class for sampling strategies."""

    def should_store(self, buffer_size: int, total_seen: int, max_size: int) -> bool:
        """Determine if we should store the current sample."""
        raise NotImplementedError

    def get_storage_index(
        self, buffer_size: int, total_seen: int, max_size: int
    ) -> int | None:
        """Get the index where to store the sample, or None if shouldn't store."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset strategy state."""
        pass


class CyclicStrategy(SamplingStrategy):
    """
    Circular buffer strategy - maintains last N samples.

    Indexing behavior:
    - get(0): oldest item in buffer (when full)
    - get(-1): newest item in buffer
    - get(-x): x-th newest item

    Note: When buffer is not full, get(0) is the first item inserted.
    """

    def __init__(self):
        self.head = 0  # Points to next insertion slot

    def should_store(self, buffer_size: int, total_seen: int, max_size: int) -> bool:
        return True  # Always store in cyclic mode

    def get_storage_index(
        self, buffer_size: int, total_seen: int, max_size: int
    ) -> int | None:
        if buffer_size < max_size:
            return buffer_size  # Still filling up
        else:
            idx = self.head
            self.head = (self.head + 1) % max_size
            return idx

    def get_logical_index(
        self, requested_index: int, buffer_size: int, max_size: int
    ) -> int:
        """Convert logical index to physical storage index."""
        if buffer_size < max_size:
            # Buffer not full - simple indexing
            if requested_index < 0:
                requested_index = buffer_size + requested_index
            return max(0, min(requested_index, buffer_size - 1))
        else:
            # Buffer full - circular indexing
            physical_index = (self.head + requested_index) % max_size
            return physical_index

    def get_recent_indices(
        self, buffer_size: int, max_size: int, n_items: int
    ) -> List[int]:
        """Get indices for the most recent n_items."""
        n_items = min(n_items, buffer_size)
        if buffer_size < max_size:
            # Buffer not full - take last n_items
            return list(range(buffer_size - n_items, buffer_size))
        else:
            # Buffer full - take last n_items from circular buffer
            indices = []
            for i in range(n_items):
                logical_idx = -(n_items - i)  # -n_items, -(n_items-1), ..., -1
                physical_idx = self.get_logical_index(
                    logical_idx, buffer_size, max_size
                )
                indices.append(physical_idx)
            return indices

    def reset(self) -> None:
        self.head = 0


class FixedStrategy(SamplingStrategy):
    """
    Fixed size strategy - stores first N samples then stops.

    Indexing behavior:
    - get(0): first item stored
    - get(-1): last item stored
    - get(-x): x-th from last item stored

    Simple, predictable indexing like a regular list.
    """

    def should_store(self, buffer_size: int, total_seen: int, max_size: int) -> bool:
        return buffer_size < max_size

    def get_storage_index(
        self, buffer_size: int, total_seen: int, max_size: int
    ) -> int | None:
        return (
            buffer_size
            if self.should_store(buffer_size, total_seen, max_size)
            else None
        )

    def get_logical_index(
        self, requested_index: int, buffer_size: int, max_size: int
    ) -> int:
        """Convert logical index to physical storage index."""
        if requested_index < 0:
            requested_index = buffer_size + requested_index
        return max(0, min(requested_index, buffer_size - 1))

    def get_recent_indices(
        self, buffer_size: int, max_size: int, n_items: int
    ) -> List[int]:
        """Get indices for the most recent n_items."""
        n_items = min(n_items, buffer_size)
        return list(range(buffer_size - n_items, buffer_size))

    def reset(self) -> None:
        pass


class UnlimitedStrategy(SamplingStrategy):
    """
    Unlimited storage strategy - stores all samples without size limit.

    Indexing behavior:
    - get(0): first item stored
    - get(-1): last item stored
    - get(-x): x-th from last item stored

    Similar to fixed strategy but without size restrictions.
    """

    def should_store(self, buffer_size: int, total_seen: int, max_size: int) -> bool:
        return True  # Always store

    def get_storage_index(
        self, buffer_size: int, total_seen: int, max_size: int
    ) -> int | None:
        return buffer_size  # Always append to end

    def get_logical_index(
        self, requested_index: int, buffer_size: int, max_size: int
    ) -> int:
        """Convert logical index to physical storage index."""
        if requested_index < 0:
            requested_index = buffer_size + requested_index
        return max(0, min(requested_index, buffer_size - 1))

    def get_recent_indices(
        self, buffer_size: int, max_size: int, n_items: int
    ) -> List[int]:
        """Get indices for the most recent n_items."""
        n_items = min(n_items, buffer_size)
        return list(range(buffer_size - n_items, buffer_size))

    def reset(self) -> None:
        pass


class ReservoirStrategy(SamplingStrategy):
    """
    Reservoir sampling strategy - probabilistic representative sampling.

    Indexing behavior:
    - get(0): some stored item (order not guaranteed)
    - get(-1): some stored item (order not guaranteed)
    - get(-x): some stored item (order not guaranteed)

    Warning: Order is not meaningful with reservoir sampling!
    Use only when you don't care about temporal order.
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)

    def should_store(self, buffer_size: int, total_seen: int, max_size: int) -> bool:
        if buffer_size < max_size:
            return True  # Still filling reservoir

        # Probability = max_size / (total_seen + 1)
        return self.rng.random() < (max_size / (total_seen + 1))

    def get_storage_index(
        self, buffer_size: int, total_seen: int, max_size: int
    ) -> int | None:
        if not self.should_store(buffer_size, total_seen, max_size):
            return None

        if buffer_size < max_size:
            return buffer_size  # Still filling
        else:
            return self.rng.randint(0, max_size)  # Random replacement

    def get_logical_index(
        self, requested_index: int, buffer_size: int, max_size: int
    ) -> int:
        """Convert logical index to physical storage index."""
        # For reservoir sampling, order doesn't matter - just clamp to valid range
        if requested_index < 0:
            requested_index = buffer_size + requested_index
        return max(0, min(requested_index, buffer_size - 1))

    def get_recent_indices(
        self, buffer_size: int, max_size: int, n_items: int
    ) -> List[int]:
        """Get indices for recent items (order not meaningful for reservoir)."""
        n_items = min(n_items, buffer_size)
        return list(range(min(n_items, buffer_size)))

    def reset(self) -> None:
        pass


class DataBuffer:
    """
    High-performance buffer for neural network data with flexible sizing.

    Thread Safety:
    - self._lock: Ensures thread-safe operations for multi-GPU distributed training

    Indexing Behavior:
    - Depends on the sampling strategy used
    - CyclicStrategy: Maintains temporal order (newest/oldest semantics)
    - FixedStrategy: Simple list-like indexing (predictable order)
    - UnlimitedStrategy: Stores all data without size limit
    - ReservoirStrategy: Order not guaranteed (for statistical sampling)

    Memory Management:
    - Explicit cleanup to prevent memory leaks
    - Efficient tensor operations with minimal copying
    - Support for unlimited growth when max_size < 0
    """

    __slots__ = [
        "max_size",
        "strategy_name",
        "cpu_offload",
        "detach_tensors",
        "thread_safe",
        "_storage",
        "_strategy",
        "_size",
        "_total_seen",
        "_lock",
        "name",
        "_unlimited",
    ]

    def __init__(
        self,
        max_size: int,
        strategy: str = "cyclic",
        cpu_offload: bool = True,
        detach_tensors: bool = True,
        thread_safe: bool = True,
        name: str = "DataBuffer",
    ):
        """
        Initialize data buffer.

        Args:
            max_size: Maximum number of samples to store, or -1 for unlimited
            strategy: Sampling strategy ("cyclic", "fixed", "reservoir", "unlimited")
            cpu_offload: Move data to CPU to save GPU memory
            detach_tensors: Detach tensors from computation graph
            thread_safe: Use thread-safe operations (important for distributed training)
            name: Buffer name for debugging
        """
        # Handle unlimited storage
        if max_size < 0 or strategy == "unlimited":
            self._unlimited = True
            self.max_size = -1
            self.strategy_name = "unlimited"
            self._storage: List[Any] = []
        else:
            self._unlimited = False
            self.max_size = max_size
            self.strategy_name = strategy
            # Pre-allocate storage for efficiency
            self._storage: List[Any] = [None] * max_size if max_size > 0 else []

        self.cpu_offload = cpu_offload
        self.detach_tensors = detach_tensors
        self.thread_safe = thread_safe
        self.name = name

        self._strategy = self._create_strategy(self.strategy_name)
        self._size = 0
        self._total_seen = 0

        # Thread safety for distributed training
        self._lock = threading.RLock() if thread_safe else None

        logger.debug(
            f"Created {self.name}: max_size={max_size}, strategy={self.strategy_name}, "
            f"thread_safe={thread_safe}, unlimited={self._unlimited}"
        )

    def _create_strategy(self, strategy: str) -> SamplingStrategy:
        """Create sampling strategy with validation."""
        strategies = {
            "cyclic": CyclicStrategy,
            "fixed": FixedStrategy,
            "reservoir": ReservoirStrategy,
            "unlimited": UnlimitedStrategy,
        }

        if strategy not in strategies:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Available: {list(strategies.keys())}"
            )

        return strategies[strategy]()

    def _preprocess_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Enhanced preprocessing with immediate GPU memory release."""
        # Detach from computation graph
        if self.detach_tensors and tensor.requires_grad:
            tensor = tensor.detach()

        # Move to CPU if requested with immediate GPU cleanup
        if self.cpu_offload and tensor.device.type != "cpu":
            # Clone to CPU and immediately clear GPU reference
            cpu_tensor = tensor.cpu()

            # Force immediate GPU memory release
            del tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            tensor = cpu_tensor

        # Ensure contiguous memory layout
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        return tensor

    def _preprocess_data(self, data: Any) -> Any:
        """Recursively preprocess data."""
        if isinstance(data, torch.Tensor):
            return self._preprocess_tensor(data)
        elif isinstance(data, dict):
            return {k: self._preprocess_data(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self._preprocess_data(item) for item in data)
        else:
            return data

    def should_store(self) -> bool:
        """Determine if the current data should be stored based on the strategy."""
        effective_max_size = len(self._storage) if self._unlimited else self.max_size
        return self._strategy.should_store(
            self._size, self._total_seen, effective_max_size
        )

    def append(self, data: Any) -> bool:
        """
        Append data to buffer.

        Args:
            data: Data to store (tensor, dict, list, etc.)

        Returns:
            bool: True if data was stored, False if rejected by strategy or max_size=0
        """
        # Quick check for zero-size buffer
        if self.max_size == 0:
            return False

        if self._lock:
            with self._lock:
                return self._append_impl(data)
        else:
            return self._append_impl(data)

    def _append_impl(self, data: Any) -> bool:
        """Internal append implementation."""
        # For unlimited storage, expand storage as needed
        if self._unlimited:
            effective_max_size = len(self._storage)
        else:
            effective_max_size = self.max_size

        # Check if strategy wants to store this sample
        if not self.should_store():
            self._total_seen += 1
            return False

        # Get storage index from strategy
        storage_idx = self._strategy.get_storage_index(
            self._size, self._total_seen, effective_max_size
        )
        if storage_idx is None:
            self._total_seen += 1
            return False

        # Preprocess data
        try:
            processed_data = self._preprocess_data(data)
        except Exception as e:
            logger.warning(f"Data preprocessing failed: {e}")
            self._total_seen += 1
            return False

        # For unlimited storage, expand list if needed
        if self._unlimited:
            while len(self._storage) <= storage_idx:
                self._storage.append(None)

        # Store data with explicit cleanup of old data
        if storage_idx < len(self._storage) and self._storage[storage_idx] is not None:
            # Clear old reference to prevent memory leaks
            self._storage[storage_idx] = None

        self._storage[storage_idx] = processed_data

        # Update size only if we're expanding
        if storage_idx >= self._size:
            self._size = storage_idx + 1

        self._total_seen += 1
        return True

    def get(self, index: int) -> Any:
        """
        Get item at logical index.

        Indexing behavior depends on strategy:
        - Cyclic: get(0)=oldest, get(-1)=newest
        - Fixed/Unlimited: get(0)=first stored, get(-1)=last stored
        - Reservoir: order not meaningful

        Args:
            index: Logical index (supports negative indexing)

        Returns:
            Stored item at the logical index
        """
        if self.max_size == 0:
            raise IndexError("Buffer has zero capacity")

        if self._lock:
            with self._lock:
                return self._get_impl(index)
        else:
            return self._get_impl(index)

    def _get_impl(self, index: int) -> Any:
        """Internal get implementation."""
        if self._size == 0:
            raise IndexError("Buffer is empty")

        # Convert logical index to physical storage index
        effective_max_size = len(self._storage) if self._unlimited else self.max_size
        physical_index = self._strategy.get_logical_index(
            index, self._size, effective_max_size
        )

        if physical_index < 0 or physical_index >= self._size:
            raise IndexError(
                f"Index {index} out of range for buffer size {self._size}"
            )

        return self._storage[physical_index]

    def get_all(self) -> List[Any]:
        """
        Get all stored data in logical order.

        For cyclic strategy: returns oldest to newest
        For fixed/unlimited strategy: returns first stored to last stored
        For reservoir strategy: arbitrary order
        """
        if self.max_size == 0:
            return []

        if self._lock:
            with self._lock:
                return self._get_all_impl()
        else:
            return self._get_all_impl()

    def _get_all_impl(self) -> List[Any]:
        """Internal get all implementation."""
        if self._size == 0:
            return []

        result = []
        effective_max_size = len(self._storage) if self._unlimited else self.max_size

        for i in range(self._size):
            try:
                physical_index = self._strategy.get_logical_index(
                    i, self._size, effective_max_size
                )
                if (
                    physical_index < len(self._storage)
                    and self._storage[physical_index] is not None
                ):
                    result.append(self._storage[physical_index])
            except IndexError:
                break

        return result

    def get_recent_items(self, n_items: int) -> List[Any]:
        """
        Get the most recent n_items from the buffer.

        Args:
            n_items: Number of recent items to retrieve

        Returns:
            List of recent items in chronological order (oldest to newest)
        """
        if self.max_size == 0 or self._size == 0:
            return []

        if self._lock:
            with self._lock:
                return self._get_recent_items_impl(n_items)
        else:
            return self._get_recent_items_impl(n_items)

    def _get_recent_items_impl(self, n_items: int) -> List[Any]:
        """Internal implementation for getting recent items."""
        effective_max_size = len(self._storage) if self._unlimited else self.max_size
        indices = self._strategy.get_recent_indices(
            self._size, effective_max_size, n_items
        )

        result = []
        for idx in indices:
            if idx < len(self._storage) and self._storage[idx] is not None:
                result.append(self._storage[idx])

        return result

    def clear(self) -> None:
        """Clear buffer with explicit memory cleanup."""
        if self._lock:
            with self._lock:
                self._clear_impl()
        else:
            self._clear_impl()

    def _clear_impl(self) -> None:
        """Internal clear implementation."""
        # Explicitly clear all storage slots to prevent memory leaks
        for i in range(len(self._storage)):
            self._storage[i] = None

        # For unlimited storage, also reset the list
        if self._unlimited:
            self._storage = []

        # Reset state
        self._size = 0
        self._total_seen = 0
        self._strategy.reset()

        # Clear GPU memory if offloading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Force garbage collection
        gc.collect()

    def to_tensor(self, dim: int = 0) -> torch.Tensor:
        """
        Concatenate all tensor data efficiently.

        Args:
            dim: Dimension along which to concatenate

        Returns:
            Concatenated tensor
        """
        if self.max_size == 0:
            raise ValueError("Buffer has zero capacity")

        data = self.get_all()
        if not data:
            raise ValueError("Buffer is empty")

        if not isinstance(data[0], torch.Tensor):
            raise ValueError("Buffer does not contain tensor data")

        valid_tensors = [item for item in data if isinstance(item, torch.Tensor)]
        if not valid_tensors:
            raise ValueError("No valid tensors in buffer")

        return torch.cat(valid_tensors, dim=dim)

    def to_dict(self, dim: int = 0) -> Dict[str, torch.Tensor]:
        """
        Convert dict data to concatenated tensors by key.

        Args:
            dim: Dimension along which to concatenate

        Returns:
            Dict with concatenated tensors for each key
        """
        if self.max_size == 0:
            return {}

        data = self.get_all()
        if not data or not isinstance(data[0], dict):
            return {}

        # Collect all unique keys across all samples
        all_keys = set()
        for item in data:
            if isinstance(item, dict):
                all_keys.update(item.keys())

        # Concatenate tensors for each key
        result = {}
        for key in all_keys:
            tensors = []
            for item in data:
                if (
                    isinstance(item, dict)
                    and key in item
                    and item[key] is not None
                    and isinstance(item[key], torch.Tensor)
                ):
                    tensors.append(item[key])

            if tensors:
                try:
                    result[key] = torch.cat(tensors, dim=dim)
                except RuntimeError as e:
                    logger.warning(
                        f"Failed to concatenate tensors for key '{key}': {e}"
                    )

        return result

    def get_storage_size(self, unit="GB") -> float:
        """Return the current size of the storage in specified unit (GB, MB, or bytes)."""
        total_bytes = 0
        storage_to_check = self._storage[: self._size] if self._size > 0 else []

        for item in storage_to_check:
            if isinstance(item, torch.Tensor):
                total_bytes += item.element_size() * item.nelement()
            elif isinstance(item, dict):
                for v in item.values():
                    if isinstance(v, torch.Tensor):
                        total_bytes += v.element_size() * v.nelement()
            elif hasattr(item, "__sizeof__"):
                total_bytes += item.__sizeof__()

        unit = unit.upper()
        if unit == "GB":
            return total_bytes / (1024**3)
        elif unit == "MB":
            return total_bytes / (1024**2)
        elif unit == "BYTES":
            return float(total_bytes)
        else:
            raise ValueError(f"Unknown unit '{unit}'. Use 'GB', 'MB', or 'BYTES'.")

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._size > 0

    def __getitem__(self, index: int) -> Any:
        return self.get(index)

    def __repr__(self) -> str:
        size_str = (
            f"{self._size}/unlimited"
            if self._unlimited
            else f"{self._size}/{self.max_size}"
        )
        return (
            f"DataBuffer(name='{self.name}', size={size_str}, "
            f"strategy={self.strategy_name})"
        )


@dataclass
class Record:
    """
    Combined storage for model records.

    Contains data that varies per batch sample and timestep:
    - guess_indices: Model predictions (batch_size, n_timesteps)
    - label_indices: Ground truth labels (batch_size, n_timesteps)
    - image_indices: Unique identifiers for input images (batch_size, n_timesteps)

    Metadata like sample indices and times indices are generated when needed.
    """

    guess_indices: torch.Tensor
    label_indices: torch.Tensor
    image_indices: torch.Tensor

    def to_cpu(self) -> "Record":
        """Move all tensors to CPU."""
        return Record(
            guess_indices=self.guess_indices.cpu(),
            label_indices=self.label_indices.cpu(),
            image_indices=self.image_indices.cpu(),
        )

    def detach(self) -> "Record":
        """Detach all tensors from computation graph."""
        return Record(
            guess_indices=self.guess_indices.detach(),
            label_indices=self.label_indices.detach(),
            image_indices=self.image_indices.detach(),
        )


class StorageBuffer:
    """
    High-level buffer management for DynVision with flexible strategy alignment.

    Combines response and record storage with efficient operations.
    Supports different buffer sizes and strategies with intelligent alignment.

    Usage:
        # Different sizes and strategies are now supported
        self.storage = StorageBuffer(
            max_responses=100, response_strategy="cyclic",
            max_records=1000, record_strategy="unlimited"
        )

        # Alignment happens automatically in get_dataframe()
        df = self.storage.get_dataframe()
    """

    def __init__(
        self,
        max_responses: int = 100,
        max_records: int = 100,
        response_strategy: str = "cyclic",
        record_strategy: str = "fixed",
        cpu_offload: bool = True,
        thread_safe: bool = True,
    ):
        # Response buffer for layer activations
        self.responses = DataBuffer(
            max_size=max_responses,
            strategy=response_strategy,
            cpu_offload=cpu_offload,
            thread_safe=thread_safe,
            name="ResponseBuffer",
        )

        # Record buffer for model predictions and metadata
        self.records = DataBuffer(
            max_size=max_records,
            strategy=record_strategy,
            cpu_offload=cpu_offload,
            thread_safe=thread_safe,
            name="RecordBuffer",
        )

        logger.debug(
            f"Storage buffers created: responses={max_responses}({response_strategy}), "
            f"records={max_records}({record_strategy})"
        )

    def store_responses(
        self, response_dict: Dict[str, torch.Tensor], precision: int = 16
    ) -> bool:
        """
        Store neural network layer responses.

        Args:
            response_dict: Dictionary of layer name to response tensor
            precision: Bit precision to store tensors (16 or 32)

        Returns:
            bool: True if data was stored, False otherwise
        """
        # Convert tensors to specified precision
        processed_dict = {}
        for key, tensor in response_dict.items():
            if precision == 16 and tensor.dtype != torch.float16:
                processed_dict[key] = tensor.to(dtype=torch.float16)
            elif precision == 32 and tensor.dtype != torch.float32:
                processed_dict[key] = tensor.to(dtype=torch.float32)
            else:
                processed_dict[key] = tensor

        return self.responses.append(processed_dict)

    def store_records(
        self,
        guess_indices: torch.Tensor,
        label_indices: torch.Tensor,
        image_indices: Optional[torch.Tensor] = None,
    ) -> bool:
        """
        Store model records.

        Args:
            guess_indices: Model predictions (batch_size, n_timesteps)
            label_indices: Ground truth labels (batch_size, n_timesteps)
            image_indices: Unique image identifiers (batch_size, n_timesteps)
        """
        if not self.records.should_store():
            self.records._total_seen += 1
            return False

        if image_indices is None:
            batch_size, n_timesteps = label_indices.shape
            image_indices = (
                torch.arange(batch_size).unsqueeze(1).expand(-1, n_timesteps)
            )

        record = Record(
            guess_indices=guess_indices,
            label_indices=label_indices,
            image_indices=image_indices,
        )

        # Apply preprocessing
        if self.records.cpu_offload:
            record = record.to_cpu()
        if self.records.detach_tensors:
            record = record.detach()

        return self.records.append(record)

    def _align_data(
        self, response_data: List[Any], record_data: List[Any]
    ) -> Tuple[List[Any], List[Any]]:
        """
        Align response and record data when using different strategies.

        This function handles the complex alignment between different sampling strategies,
        taking into account their temporal characteristics and indexing behavior.

        Strategy temporal characteristics:
        - Fixed: Chronological samples [0, N), stops when full
        - Unlimited: All chronological samples [0, M)
        - Cyclic: Most recent N samples, maintains temporal order in get_all()
        - Reservoir: Random samples, no temporal meaning

        Args:
            response_data: List of response dictionaries from get_all()
            record_data: List of record objects from get_all()

        Returns:
            Tuple of (aligned_responses, aligned_records)
        """
        if not response_data or not record_data:
            return [], []

        resp_strategy = self.responses.strategy_name
        rec_strategy = self.records.strategy_name
        min_length = min(len(response_data), len(record_data))

        # Same strategy: straightforward alignment
        if resp_strategy == rec_strategy:
            if resp_strategy in ["cyclic"]:
                # Both cyclic: both contain most recent samples in chronological order
                # Take the overlap of most recent samples
                return response_data[-min_length:], record_data[-min_length:]
            else:
                # Both fixed/unlimited: both contain chronological samples from start
                # Take the overlap from the beginning
                return response_data[:min_length], record_data[:min_length]

        # Mixed strategies: need careful alignment
        return self._align_mixed_strategies(
            response_data, record_data, resp_strategy, rec_strategy, min_length
        )

    def _align_mixed_strategies(
        self,
        response_data: List[Any],
        record_data: List[Any],
        resp_strategy: str,
        rec_strategy: str,
        min_length: int,
    ) -> Tuple[List[Any], List[Any]]:
        """
        Handle alignment between different strategy combinations.

        Key insight: We need to understand what samples each buffer actually contains
        relative to the total sequence of samples processed.

        Assumptions for simultaneous buffer filling:
        - Fixed buffer contains samples [0, N) if total_seen >= N, else [0, total_seen)
        - Cyclic buffer contains samples [max(0, total_seen-N), total_seen)
        - Unlimited buffer contains samples [0, total_seen)
        """

        # Get buffer metadata for smarter alignment
        resp_total_seen = getattr(self.responses, "_total_seen", len(response_data))
        rec_total_seen = getattr(self.records, "_total_seen", len(record_data))
        resp_max_size = (
            self.responses.max_size if self.responses.max_size > 0 else float("inf")
        )
        rec_max_size = (
            self.records.max_size if self.records.max_size > 0 else float("inf")
        )

        logger.debug(
            f"Aligning {resp_strategy}(seen={resp_total_seen}, max={resp_max_size}) "
            f"with {rec_strategy}(seen={rec_total_seen}, max={rec_max_size})"
        )

        # Strategy combination handling
        if resp_strategy == "fixed" and rec_strategy == "unlimited":
            # Fixed: [0, min(resp_max_size, resp_total_seen))
            # Unlimited: [0, rec_total_seen)
            # Overlap: [0, min(len(response_data), len(record_data)))
            return response_data[:min_length], record_data[:min_length]

        elif resp_strategy == "unlimited" and rec_strategy == "fixed":
            # Symmetric case
            return response_data[:min_length], record_data[:min_length]

        elif resp_strategy == "fixed" and rec_strategy == "cyclic":
            # Fixed: [0, min(resp_max_size, resp_total_seen))
            # Cyclic: [max(0, rec_total_seen - rec_max_size), rec_total_seen)

            # Check if there's temporal overlap
            fixed_end = min(resp_max_size, resp_total_seen)
            cyclic_start = max(0, rec_total_seen - rec_max_size)

            if fixed_end <= cyclic_start:
                # No temporal overlap - warn user
                logger.warning(
                    f"No temporal overlap between fixed buffer [0, {fixed_end}) "
                    f"and cyclic buffer [{cyclic_start}, {rec_total_seen}). "
                    f"Taking most recent data from each buffer for analysis."
                )
                # Take last samples from fixed, recent samples from cyclic
                return response_data[-min_length:], record_data[-min_length:]
            else:
                # There is overlap - try to align overlapping period
                # This is complex without sample timestamps, so fall back to recent data
                logger.info(
                    f"Partial temporal overlap detected. Taking recent data from each buffer."
                )
                return response_data[-min_length:], record_data[-min_length:]

        elif resp_strategy == "cyclic" and rec_strategy == "fixed":
            # Symmetric case of above
            fixed_end = min(rec_max_size, rec_total_seen)
            cyclic_start = max(0, resp_total_seen - resp_max_size)

            if fixed_end <= cyclic_start:
                logger.warning(
                    f"No temporal overlap between cyclic buffer [{cyclic_start}, {resp_total_seen}) "
                    f"and fixed buffer [0, {fixed_end}). "
                    f"Taking most recent data from each buffer for analysis."
                )
                return response_data[-min_length:], record_data[-min_length:]
            else:
                logger.info(
                    f"Partial temporal overlap detected. Taking recent data from each buffer."
                )
                return response_data[-min_length:], record_data[-min_length:]

        elif resp_strategy == "cyclic" and rec_strategy == "unlimited":
            # Cyclic: most recent samples
            # Unlimited: all samples, so take the most recent to match cyclic
            return response_data[-min_length:], record_data[-min_length:]

        elif resp_strategy == "unlimited" and rec_strategy == "cyclic":
            # Symmetric case
            return response_data[-min_length:], record_data[-min_length:]

        elif resp_strategy == "unlimited" and rec_strategy == "unlimited":
            # Both unlimited: should contain the same samples if filled simultaneously
            # Take from beginning to get the full overlap (most data)
            # Or take from end if we prefer most recent data
            if min_length == len(response_data) == len(record_data):
                # Same length - they should be identical sequences
                return response_data, record_data
            else:
                # Different lengths - take overlapping portion from the beginning
                return response_data[:min_length], record_data[:min_length]

        else:
            # Handle reservoir or unknown strategies
            logger.warning(
                f"Alignment between '{resp_strategy}' and '{rec_strategy}' may not be meaningful. "
                f"Taking first {min_length} samples from each buffer."
            )
            return response_data[:min_length], record_data[:min_length]

    def get_dataframe(self, layer_name: str = "classifier") -> pd.DataFrame:
        """
        Generate classifier DataFrame efficiently with automatic strategy alignment.

        Now supports different buffer sizes and strategies by intelligently aligning the data.
        The alignment logic prioritizes data recency and logical sample correspondence.
        """

        try:
            # Quick check for zero-capacity buffers
            if self.responses.max_size == 0 or self.records.max_size == 0:
                return pd.DataFrame()

            # Get raw data from buffers
            response_data = self.responses.get_all()
            record_data = self.records.get_all()

            # Log format and shapes for debugging
            logger.info(
                f"Response data format: {type(response_data)}, length: {len(response_data)}"
            )
            if response_data:
                logger.info(f"First response item type: {type(response_data[0])}")
                if isinstance(response_data[0], dict):
                    for key, value in response_data[0].items():
                        if isinstance(value, torch.Tensor):
                            logger.info(f"  {key}: {value.shape}")
                        else:
                            logger.info(f"  {key}: {type(value)}")

            logger.info(
                f"Record data format: {type(record_data)}, length: {len(record_data)}"
            )
            if record_data:
                logger.info(f"First record item type: {type(record_data[0])}")
                if hasattr(record_data[0], "guess_indices"):
                    logger.info(
                        f"  guess_indices: {record_data[0].guess_indices.shape}"
                    )
                    logger.info(
                        f"  label_indices: {record_data[0].label_indices.shape}"
                    )
                    logger.info(
                        f"  image_indices: {record_data[0].image_indices.shape}"
                    )

            if not response_data or not record_data:
                logger.warning("No data stored in buffers")
                return pd.DataFrame()

            # Align data using different strategies
            response_data, record_data = self._align_data(response_data, record_data)

            if not response_data or not record_data:
                logger.warning("No aligned data available")
                return pd.DataFrame()

            logger.info(
                f"Aligned data: {len(response_data)} responses, {len(record_data)} records "
                f"(strategies: {self.responses.strategy_name}, {self.records.strategy_name})"
            )

            # Check if the layer exists in responses
            if layer_name not in response_data[0]:
                available_layers = (
                    list(response_data[0].keys()) if response_data else []
                )
                logger.warning(
                    f"Layer '{layer_name}' not found. Available: {available_layers}"
                )
                return pd.DataFrame()

            # Extract response tensors and pad them to the same time step length
            max_timesteps = max(item[layer_name].shape[1] for item in response_data)
            response_tensors = []
            for item in response_data:
                tensor = item[layer_name]
                pad_len = max_timesteps - tensor.shape[1]
                if pad_len > 0:
                    # Pad at the start along axis 1 (time steps) with zeros
                    pad = (0, 0, 0, 0, 0, 0, pad_len, 0)
                    tensor = torch.nn.functional.pad(
                        tensor, pad, mode="constant", value=0
                    )
                response_tensors.append(tensor)

            layer_responses = torch.cat(response_tensors, dim=0)

            # Extract and concatenate records
            guess_tensors = [record.guess_indices for record in record_data]
            guess_data = torch.cat(guess_tensors, dim=0)

            label_tensors = [record.label_indices for record in record_data]
            label_data = torch.cat(label_tensors, dim=0)

            image_tensors = [record.image_indices for record in record_data]
            image_data = torch.cat(image_tensors, dim=0)

            # Ensure all data has the same length (number of samples)
            valid_data_length = min(
                len(layer_responses),
                len(guess_data),
                len(label_data),
                len(image_data),
            )

            layer_responses = layer_responses[:valid_data_length]
            guess_data = guess_data[:valid_data_length]
            label_data = label_data[:valid_data_length]
            image_data = image_data[:valid_data_length]

            # Convert to CPU and numpy
            response = layer_responses.cpu().float().numpy()
            label_indices = label_data.cpu().numpy()
            guess_indices = guess_data.cpu().numpy()
            image_indices = image_data.cpu().numpy()

            # Get dimensions
            n_samples, n_timesteps, n_classes = response.shape

            # Create indices for DataFrame structure
            sample_indices, times_indices, class_indices = np.meshgrid(
                np.arange(n_samples),
                np.arange(n_timesteps),
                np.arange(n_classes),
                indexing="ij",
            )
            # label_sets = np.array(["".join(row.astype(str)) for row in label_indices])
            # label_sets = (
            #     label_sets[:, None, None]
            #     .repeat(n_timesteps, axis=-2)
            #     .repeat(n_classes, axis=-1)
            # )
            label_indices = label_indices[..., None].repeat(n_classes, axis=-1)
            guess_indices = guess_indices[..., None].repeat(n_classes, axis=-1)
            image_indices = image_indices[..., None].repeat(n_classes, axis=-1)

            df = pd.DataFrame(
                {
                    "sample_index": sample_indices.ravel(),
                    "times_index": times_indices.ravel(),
                    "class_index": class_indices.ravel(),
                    "response": response.ravel(),
                    "label_index": label_indices.ravel(),
                    "guess_index": guess_indices.ravel(),
                    "image_index": image_indices.ravel(),
                    # "label_set": label_sets.ravel(),
                }
            )

            # Clean up memory
            del (
                response,
                label_indices,
                guess_indices,
                image_indices,
                sample_indices,
                times_indices,
                class_indices,
            )

            return df

        except Exception as e:
            logger.error(f"Error generating classifier DataFrame: {e}")
            return pd.DataFrame()

    def clear_all(self) -> None:
        """Clear all buffers."""
        self.responses.clear()
        self.records.clear()

    def get_storage_size(self, unit="GB") -> float:
        """Get the size of the storage."""
        return self.responses.get_storage_size(unit) + self.records.get_storage_size(
            unit
        )

    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage information."""
        return {
            "responses_size": len(self.responses),
            "records_size": len(self.records),
            "total_items": len(self.responses) + len(self.records),
            "responses_strategy": self.responses.strategy_name,
            "records_strategy": self.records.strategy_name,
        }


class StorageBufferMixin(LightningModule):
    """
    Mixin class for automatic StorageBuffer lifecycle management in PyTorch Lightning.

    Now supports different buffer configurations with automatic alignment in get_dataframe().
    You can specify different sizes and strategies for responses vs records.
    """

    # Default storage configurations - can use different strategies
    training_storage_config: Dict[str, Any] = {
        "max_responses": 0,  # Disabled by default
        "max_records": 0,  # Disabled by default
        "response_strategy": "fixed",
        "record_strategy": "fixed",
        "cpu_offload": True,
        "thread_safe": True,
    }

    validation_storage_config: Dict[str, Any] = {
        "max_responses": 0,  # Enabled by default
        "max_records": 0,  # Enabled by default
        "response_strategy": "fixed",
        "record_strategy": "fixed",
        "cpu_offload": True,
        "thread_safe": True,
    }

    testing_storage_config: Dict[str, Any] = {
        "max_responses": 5,  # Enabled by default for analysis
        "max_records": 5,  # Much larger for detailed analysis
        "response_strategy": "fixed",  # Keep recent responses
        "record_strategy": "fixed",  # Keep all records
        "cpu_offload": True,
        "thread_safe": True,
    }

    def __init__(
        self,
        store_train_responses: Optional[int] = None,
        store_val_responses: Optional[int] = None,
        store_test_responses: Optional[int] = None,
        store_train_records: Optional[int] = None,
        store_val_records: Optional[int] = None,
        store_test_records: Optional[int] = None,
        early_test_stop: bool = True,
        **kwargs,
    ):
        """Initialize with flexible storage configuration."""
        super().__init__(**kwargs)

        # Update response configs only if provided
        if store_train_responses is not None:
            self.training_storage_config["max_responses"] = store_train_responses
        if store_val_responses is not None:
            self.validation_storage_config["max_responses"] = store_val_responses
        if store_test_responses is not None:
            self.testing_storage_config["max_responses"] = store_test_responses

        # Update record configs only if provided
        if store_train_records is not None:
            self.training_storage_config["max_records"] = store_train_records
        if store_val_records is not None:
            self.validation_storage_config["max_records"] = store_val_records
        if store_test_records is not None:
            self.testing_storage_config["max_records"] = store_test_records

        # Set early stopping configuration
        self.early_test_stop = early_test_stop

        # Always create a storage instance (with zero capacity initially)
        self.storage = StorageBuffer(
            max_responses=0,
            max_records=0,
            cpu_offload=True,
            thread_safe=True,
        )

        # Flag to track when to stop testing early
        self._stop_testing_early = False

    def get_dataframe(self, **kwargs) -> pd.DataFrame:
        return self.storage.get_dataframe(**kwargs)

    def _check_memory_usage(self, stage: str):
        """Monitor memory usage and warn about potential issues."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            cached = torch.cuda.memory_reserved() / 1024**2  # MB

            if (
                allocated
                > 0.8 * torch.cuda.get_device_properties(0).total_memory / 1024**2
            ):
                logger.warning(
                    f"High GPU memory usage in {stage}: {allocated:.0f}MB allocated, "
                    f"{cached:.0f}MB cached"
                )

                # Emergency buffer reduction
                if hasattr(self.storage, "clear_all"):
                    logger.warning("Emergency: Clearing storage buffers")
                    self.storage.clear_all()
                    torch.cuda.empty_cache()

    def _check_buffer_filled(self) -> bool:
        """
        Check if response buffer is filled and should stop early.

        Returns:
            bool: True if testing should be stopped early
        """
        # Only check if early stopping is enabled
        if not self.early_test_stop:
            return False

        # Skip for unlimited or non-fixed strategies
        if (
            self.storage.responses.max_size <= 0
            or self.storage.responses.strategy_name not in ["fixed"]
        ):
            return False

        # Check if buffer is filled
        is_filled = len(self.storage.responses) >= self.storage.responses.max_size

        if is_filled and not self._stop_testing_early:
            logger.info(
                f"Response buffer filled ({len(self.storage.responses)}/{self.storage.responses.max_size}). "
                f"Stopping testing early."
            )
            self._stop_testing_early = True

        return is_filled

    def on_test_batch_end(self, *args, **kwargs):
        self._check_memory_usage("test")

        # Check if we should stop testing early
        if self._check_buffer_filled():
            # Signal to PyTorch Lightning to stop the test loop
            # This is done by raising a specific exception that PL handles
            from pytorch_lightning.utilities.exceptions import _TunerExitException

            raise _TunerExitException("Stopping test early: response buffer filled")

    def on_train_epoch_start(self) -> None:
        """Initialize storage buffer at start of training epoch."""
        try:
            super().on_train_epoch_start()
        except AttributeError:
            pass

        logger.info(f"Storage: {self.storage.get_storage_size('GB')} GB")
        # Create storage with training configuration
        self.storage.clear_all()
        self.storage = StorageBuffer(**self.training_storage_config)
        logger.debug(f"Training storage: {self.training_storage_config}")

    def on_train_epoch_end(self) -> None:
        """Clear storage buffer at end of training epoch."""
        try:
            super().on_train_epoch_end()
        except AttributeError:
            pass

        # Clear storage and reset to empty
        self.storage.clear_all()
        self.storage = StorageBuffer(max_responses=0, max_records=0)
        logger.debug("Training storage cleared")

    def on_validation_start(self) -> None:
        """Initialize storage buffer at start of validation epoch."""
        try:
            super().on_validation_start()
        except AttributeError:
            pass

        self.storage.clear_all()

    def on_validation_epoch_start(self) -> None:
        """Initialize storage buffer at start of validation epoch."""
        try:
            super().on_validation_epoch_start()
        except AttributeError:
            pass

        logger.info(f"Storage: {self.storage.get_storage_size('GB')} GB")
        self.storage.clear_all()
        # Create storage with validation configuration
        self.storage = StorageBuffer(**self.validation_storage_config)
        logger.debug(f"Validation storage: {self.validation_storage_config}")

    def on_validation_epoch_end(self) -> None:
        """Clear storage buffer at end of validation."""
        try:
            super().on_validation_epoch_end()
        except AttributeError:
            pass

        # Clear storage and reset to empty
        self.storage.clear_all()
        self.storage = StorageBuffer(max_responses=0, max_records=0)
        logger.debug("Validation storage cleared")

    def on_test_start(self) -> None:
        """Initialize storage buffer at start of testing."""
        try:
            super().on_test_start()
        except AttributeError:
            pass

        # Reset early stop flag
        self._stop_testing_early = False

        # Create storage with testing configuration
        self.storage.clear_all()
        self.storage = StorageBuffer(**self.testing_storage_config)
        logger.debug(f"Testing storage: {self.testing_storage_config}")

    def on_test_end(self) -> None:
        """Clear storage buffer at end of testing."""
        try:
            super().on_test_end()
        except AttributeError:
            pass


if __name__ == "__main__":
    print("Testing enhanced DataBuffer implementation...")

    # Test unlimited strategy
    print("\n=== Testing Unlimited Strategy ===")
    unlimited_buffer = DataBuffer(max_size=-1, strategy="unlimited")
    for i in range(5):
        result = unlimited_buffer.append(torch.tensor([i]))
        print(f"Append {i}: {result}, size: {len(unlimited_buffer)}")

    print(
        f"Unlimited buffer contents: {[item.item() for item in unlimited_buffer.get_all()]}"
    )

    # Test mixed strategies alignment
    print("\n=== Testing Mixed Strategy Alignment ===")

    class TestModel(StorageBufferMixin):
        """Test model with mixed strategies."""

        testing_storage_config = {
            "max_responses": 3,  # Small response buffer
            "max_records": 10,  # Larger record buffer
            "response_strategy": "cyclic",  # Keep recent responses
            "record_strategy": "unlimited",  # Keep all records
        }

        def __init__(self):
            super().__init__()

    # Test lifecycle with mixed strategies
    model = TestModel()
    print(f"Initial storage: {model.storage.get_memory_info()}")

    # Start testing
    model.on_test_start()
    print(f"Test storage: {model.storage.get_memory_info()}")

    # Store different amounts of data
    for i in range(7):
        responses = {"classifier": torch.randn(1, 5, 4)}  # (batch, timesteps, classes)
        model.storage.store_responses(responses)

        records_stored = model.storage.store_records(
            guess_indices=torch.randint(0, 4, (1, 5)),
            label_indices=torch.randint(0, 4, (1, 5)),
        )
        print(f"  Batch {i}: stored, storage_size={model.storage.get_memory_info()}")

    # Generate DataFrame with alignment
    df = model.storage.get_dataframe()
    print(f"Generated DataFrame with mixed strategies: {df.shape}")

    if not df.empty:
        print(
            f"Sample indices range: {df['sample_index'].min()} to {df['sample_index'].max()}"
        )

    print("\nEnhanced implementation tests completed!")
