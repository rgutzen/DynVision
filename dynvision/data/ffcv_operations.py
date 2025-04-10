from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from dataclasses import replace
from typing import Callable, Optional, Tuple
from ffcv.pipeline.state import State
import numpy as np


class ExtendDataTimeFFCV(Operation):
    def __init__(self, n_timesteps=1):
        super().__init__()
        self.n_timesteps = n_timesteps

    def generate_code(self) -> Callable:
        n_timesteps = self.n_timesteps

        def extend_data_time(x, dst):
            # Determine dimensions using a method that works for NumPy and PyTorch
            n_dims = len(x.shape)

            # For batched input [batch, height, width, channels]
            if n_dims == 4:
                batch_size = x.shape[0]
                for i in range(batch_size):
                    for t in range(n_timesteps):
                        # Copy each sample to all timesteps
                        dst[i, t] = x[i]
            # For unbatched input [height, width, channels]
            elif n_dims == 3:
                for t in range(n_timesteps):
                    dst[t] = x
            else:
                # Use string interpolation that works with both NumPy and PyTorch
                raise ValueError(
                    f"Expected 3D or 4D input tensor, got shape: {x.shape}"
                )
            return dst

        return extend_data_time

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        if len(previous_state.shape) == 4:  # [batch, height, width, channels]
            batch, height, width, channels = previous_state.shape
            new_shape = (batch, self.n_timesteps, height, width, channels)
        elif len(previous_state.shape) == 3:  # [height, width, channels]
            height, width, channels = previous_state.shape
            new_shape = (self.n_timesteps, height, width, channels)
        else:
            raise ValueError(
                f"Expected 3D or 4D input tensor, got shape: {previous_state.shape}"
            )
        # Create new state with updated shape
        new_state = replace(previous_state, shape=new_shape)
        # Allocate memory for the output
        mem_allocation = AllocationQuery(
            new_shape, previous_state.dtype, device=previous_state.device
        )
        return new_state, mem_allocation


class ExtendLabelTimeFFCV(Operation):
    def __init__(self, n_timesteps=1):
        super().__init__()
        self.n_timesteps = n_timesteps

    def generate_code(self) -> Callable:
        n_timesteps = self.n_timesteps

        def extend_label_time(x, dst):
            # Determine dimensions using a method that works for both types
            n_dims = len(x.shape)

            # For 1D labels [dim]
            if n_dims == 1:
                for t in range(n_timesteps):
                    dst[t] = x
            # For 2D labels [batch, dim]
            elif n_dims == 2:
                batch_size = x.shape[0]
                for i in range(batch_size):
                    for t in range(n_timesteps):
                        dst[i, t] = x[i]
            else:
                raise ValueError(
                    f"Expected 1D or 2D input tensor, got shape: {x.shape}"
                )
            return dst

        return extend_label_time

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        if len(previous_state.shape) == 1:  # [dim]
            dim = previous_state.shape[0]
            new_shape = (self.n_timesteps, dim)
        elif len(previous_state.shape) == 2:  # [batch, dim]
            batch, dim = previous_state.shape
            new_shape = (batch, self.n_timesteps, dim)
        else:
            raise ValueError(
                f"Expected 1D or 2D input tensor, got shape: {previous_state.shape}"
            )
        # Create new state with updated shape
        new_state = replace(previous_state, shape=new_shape)
        # Allocate memory for the output
        mem_allocation = AllocationQuery(
            new_shape, previous_state.dtype, device=previous_state.device
        )
        return new_state, mem_allocation
