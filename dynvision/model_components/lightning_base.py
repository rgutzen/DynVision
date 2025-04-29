"""
Provides base classes for constructing biologically-inspired neural network models.

Classes:
    - UtilityBase (nn.Module): Core utilities for standardizing input/label dimensions,
      state dictionary management, and parameter logging.
    - LightningBase (UtilityBase, pl.LightningModule): Extends UtilityBase with PyTorch Lightning
      support for training, validation, and testing, including methods for forward propagation,
      optimizer/scheduler configuration, and evaluation.

Usage:
    Subclass LightningBase to implement custom models with advanced recurrent and feedback features.
"""

import io
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb

from dynvision import losses
from dynvision.utils import (
    alias_kwargs,
    path_to_index,
    load_config,
    on_same_device,
)
from dynvision.project_paths import project_paths

__all__ = ["UtilityBase", "LightningBase"]

defaults = SimpleNamespace(
    **load_config(project_paths.scripts.configs / "config_defaults.yaml")
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UtilityBase(nn.Module):
    """
    A base class providing utility functions for neural network models.
    Includes methods for adjusting input dimensions, managing state dictionaries,
    and handling tensor operations.
    """

    def _adjust_input_dimensions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adjust the input tensor dimensions to match the expected format.

        Args:
            x (torch.Tensor): Input tensor with dimensions in one of the following formats:
                - (dim_y, dim_x)
                - (batch_size, dim_y, dim_x)
                - (batch_size, dim_channels, dim_y, dim_x)
                - (batch_size, n_timesteps, dim_channels, dim_y, dim_x)

        Returns:
            torch.Tensor: Tensor with dimensions in the format:
                (batch_size, n_timesteps, dim_channels, dim_y, dim_x)
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 3:
            x = x.unsqueeze(1).unsqueeze(1)
        elif len(x.shape) == 4:
            x = x.unsqueeze(1)
        elif len(x.shape) == 5:
            pass
        elif len(x.shape) == 6:  # assume duplicate batch dim
            x = x[0]
        else:
            raise ValueError(
                f"Invalid input shape: {x.shape}. Expected formats: (dim_y, dim_x), (batch_size, dim_y, dim_x), (batch_size, channels, dim_y, dim_x), or (batch_size, n_timesteps, channels, dim_y, dim_x)"
            )

        return x

    def _adjust_label_dimensions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adjust the label tensor dimensions to match the expected format.

        Args:
            x (torch.Tensor): Label tensor with dimensions in one of the following formats:
                - (batch_size)
                - (batch_size, n_timesteps)

        Returns:
            torch.Tensor: Tensor with dimensions in the format:
                (batch_size, n_timesteps)
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        elif len(x.shape) == 2:
            pass
        else:
            raise ValueError(f"Invalid label shape: {x.shape}")
        return x

    def _expand_input_channels(
        self, x: Optional[torch.Tensor], n_target_channels: int = 3
    ) -> Optional[torch.Tensor]:
        """
        Expand the input tensor's channels to match the target number of channels.

        Args:
            x (Optional[torch.Tensor]): Input tensor with shape (batch_size, n_channels, ...).
            n_target_channels (int): Target number of channels.

        Returns:
            Optional[torch.Tensor]: Tensor with expanded channels or None if input is None.
        """
        if x is None:
            return None

        batch_size, n_channels, *shape = x.shape

        if n_channels == 1:
            x = x.expand(batch_size, n_target_channels, *shape)
        elif n_channels == n_target_channels:
            pass
        else:
            raise ValueError(f"Invalid number of input channels: {n_channels}")

        return x

    def _calculate_layer_output_shape(
        self,
        input_shape: Tuple[int, ...],
        kernel_size: int,
        stride: int,
        padding: int = 0,
    ) -> Tuple[int, ...]:
        """
        Calculate the output shape of a layer given its input shape and parameters.

        Args:
            input_shape (Tuple[int, ...]): Shape of the input tensor.
            kernel_size (int): Size of the kernel.
            stride (int): Stride of the operation.
            padding (int, optional): Padding applied to the input. Defaults to 0.

        Returns:
            Tuple[int, ...]: Shape of the output tensor.
        """
        return tuple(
            (dim - kernel_size + 2 * padding) // stride + 1 for dim in input_shape
        )

    def _determine_residual_timesteps(
        self, max_timesteps=100, device=None, dtype=torch.float16
    ) -> int:
        """
        Determine the number of residual timesteps required for an input to be processed through the unrolled model.

        This method uses a random input tensor to forward propagate through the model
        and checks for non-empty outputs. The process stops when a non-empty output
        is detected or a maximum of max_timesteps iterations is reached.

        Returns:
            int: Number of residual timesteps required.

        Raises:
            ValueError: If the number of residual timesteps exceeds max_timesteps.
        """
        # Get model dtype from parameters
        if dtype is None:
            dtype = next(self.parameters()).dtype
        if device is None:
            device = next(self.parameters()).device

        random_input = torch.randn(
            (1, self.n_channels, self.dim_y, self.dim_x), dtype=dtype, device=device
        )

        if hasattr(self, "reset"):
            self.reset()

        x = None
        t = -1
        is_empty_output = lambda x: (x is None) or (
            torch.all(x.eq(0)) or (torch.isnan(x).all()) or x.grad_fn is None
        )

        logger.info("Determining residual timesteps...")
        with on_same_device(
            x=x,
            **self.get_safely_named_parameters_dict(),
            label="determine_residual_timesteps",
        ):
            while is_empty_output(x):
                t += 1
                x, _ = self._forward(random_input, t=t, feedforward_only=True)
                if t > max_timesteps:
                    raise ValueError(
                        f"Unable to determine residual timesteps (> {max_timesteps})!"
                    )

        logger.info(f"Residual timesteps: {t}")

        if hasattr(self, "reset"):
            self.reset()

        return t

    def _add_missing_parameters_to_state_dict(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Add missing parameters to the model's state dictonary to the input's state dictionary.

        Args:
            state_dict (Dict[str, torch.Tensor]): State dictionary to update.

        Returns:
            Dict[str, torch.Tensor]: Updated state dictionary with missing parameters added.
        """
        missing_parameter_names = []
        for key in self.state_dict().keys():
            if key not in state_dict.keys():
                missing_parameter_names.append(key)
                state_dict[key] = self.state_dict()[key]
        if missing_parameter_names:
            logger.info(
                f"Adding missing parameters to loaded state dict: {missing_parameter_names}"
            )
        return state_dict

    def _remove_unexpected_parameters_from_state_dict(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Remove unexpected parameters from the input's state dictionary that are not found in the model's state dictionary.

        Args:
            state_dict (Dict[str, torch.Tensor]): State dictionary to update.

        Returns:
            Dict[str, torch.Tensor]: Updated state dictionary with unexpected parameters removed.
        """
        unexpected_parameter_names = []
        for key in state_dict.keys():
            if key not in self.state_dict().keys():
                unexpected_parameter_names.append(key)
        if unexpected_parameter_names:
            logger.warning(
                f"Removing unexpected parameters from loaded state dict: {unexpected_parameter_names}"
            )
            for key in unexpected_parameter_names:
                del state_dict[key]
        return state_dict

    def load_pretrained_state_dict(
        self, check_mismatch_layer: List[str] = [], strict: bool = True
    ) -> None:
        """
        Load a pretrained state dictionary into the model. This function requires the model to implement a method to download the pretrained weights 'download_pretrained_state_dict()'. If modules in the model have different names in the pretrained weights, a method 'translate_pretrained_layer_names()' should be implemented to return a mapping between the pretrained and model layer names in the form of a dictionary {"pretrained_name": "new_name"}', where pretrained_name can be any substring of a named parameter.

        Args:
            check_mismatch_layer (List[str], optional): List of layer names to check for mismatched shapes. Defaults to [].
            strict (bool, optional): Whether to strictly enforce that the keys in `state_dict` match the model's keys. Defaults to True.
        """
        # Load the pretrained weights
        if hasattr(self, "download_pretrained_state_dict"):
            state_dict = self.download_pretrained_state_dict()
        else:
            raise NotImplementedError("No method to download pretrained weights")

        # translate keys in loaded state dict
        if hasattr(self, "translate_pretrained_layer_names"):
            translate_layer_names = self.translate_pretrained_layer_names()
            external_keys = list(state_dict.keys())
            new_state_dict = copy(state_dict)
            for key in external_keys:
                for old_key, new_key in translate_layer_names.items():
                    if old_key in key:
                        new_key = key.replace(old_key, new_key)
                        new_state_dict[new_key] = state_dict[key]
                        if old_key not in translate_layer_names.values():
                            del new_state_dict[key]
                        continue
            state_dict = new_state_dict

        pretrained_keys = list(state_dict.keys())

        # adapt to different number of output classes
        for layer_name in check_mismatch_layer:
            layer_keys = [key for key in pretrained_keys if key.startswith(layer_name)]

            for key in layer_keys:
                pretrained_shape = state_dict[key].shape
                initialized_shape = self.state_dict()[key].shape

                if pretrained_shape != initialized_shape:
                    state_dict[key] = self.state_dict()[key]
                    pretrained_keys.remove(key)

        self.load_state_dict(state_dict, strict=strict)

        # select trainable parameters
        parameter_names = list(self.state_dict().keys())
        self.trainable_parameter_names = [
            p for p in parameter_names if p not in pretrained_keys
        ]

    def load_state_dict(
        self, state_dict: Dict[str, torch.Tensor], **kwargs: Any
    ) -> None:
        state_dict = self._add_missing_parameters_to_state_dict(state_dict)
        state_dict = self._remove_unexpected_parameters_from_state_dict(state_dict)
        super().load_state_dict(state_dict, **kwargs)

    def hasattr(self, attribute_name: str) -> bool:
        attributes = attribute_name.split(".")
        attr = self
        for attr_name in attributes:
            if not hasattr(attr, attr_name):
                return False
            attr = getattr(attr, attr_name)
        return True

    def getattr(self, attribute_name: str):
        attributes = attribute_name.split(".")
        attr = self
        for attr_name in attributes:
            attr = getattr(attr, attr_name)
        return attr

    def set_trainable_parameters(
        self, parameter_names: List[str] = [], force_train_all: bool = False
    ) -> None:
        if hasattr(self, "trainable_parameter_names") and len(
            self.trainable_parameter_names
        ):
            logger.warning(
                f"self.trainable_parameter_names is not empty: {self.trainable_parameter_names}! This will be overwritten!"
            )

        if not len(parameter_names):
            parameter_names = list(self.state_dict().keys())

        self.trainable_parameter_names = []
        for parameter_name in parameter_names:
            if self.hasattr(parameter_name):
                parameter = self.getattr(parameter_name)
            else:
                logger.warning(
                    f"Parameter {parameter_name} not found in model, can't be set to trainable!"
                )
                continue
            if force_train_all:
                try:
                    parameter.requires_grad = True
                except Exception as e:
                    logger.warning(
                        f"Parameter {parameter} can't be set to requires_grad=True! \n\t{e}"
                    )
            if hasattr(parameter, "requires_grad") and parameter.requires_grad:
                self.trainable_parameter_names.append(parameter_name)

    def set_trainable_parameter(self, parameter_name: str) -> None:
        if hasattr(self, parameter_name):
            if hasattr(self, "trainable_parameter_names"):
                if parameter_name not in self.trainable_parameter_names:
                    self.trainable_parameter_names.append(parameter_name)
            else:
                self.trainable_parameter_names = [parameter_name]
        else:
            logger.warning(
                f"Parameter {parameter_name} not found in model, can't be set to trainable!"
            )

    def get_trainable_parameter_names(self) -> List[str]:
        """
        Get the names of trainable parameters in the model.

        Returns:
            List[str]: List of trainable parameter names.
        """
        if not hasattr(self, "trainable_parameter_names"):
            self.set_trainable_parameters()
        return self.trainable_parameter_names

    def get_trainable_parameters(self) -> List[torch.Tensor]:
        """
        Get the trainable parameters of the model.

        Returns:
            List[torch.Tensor]: List of trainable parameters.
        """
        if not hasattr(self, "trainable_parameter_names"):
            self.set_trainable_parameters()
        return list(self.trainable_parameters())

    def trainable_parameters(self) -> torch.Generator:
        """
        Yield trainable parameters of the model. Refers to 'trainable_parameter_names' attribute if defined, else `parameters`.

        Returns:
            torch.Generator: Generator yielding trainable parameters.
        """
        if hasattr(self, "trainable_parameter_names"):
            for name, param in self.named_parameters():
                if name in self.trainable_parameter_names:
                    yield param
        else:
            yield from self.parameters()

    def named_trainable_parameters(self) -> torch.Generator:
        """
        Yield trainable parameters of the model. Refers to 'trainable_parameter_names' attribute if defined, else `named_parameters`.

        Returns:
            torch.Generator: Generator yielding tuples of parameter names and trainable parameters.
        """
        if hasattr(self, "trainable_parameter_names"):
            for name, param in self.named_parameters():
                if name in self.trainable_parameter_names:
                    yield name, param
        else:
            yield from self.named_parameters()

    def print_trainable_parameter_names(self) -> None:
        """
        Print the names of trainable and fixed parameters in the model.
        """
        if not hasattr(self, "trainable_parameter_names"):
            self.set_trainable_parameters()

        trainable, fixed = [], []
        for name, param in self.named_parameters():
            if name in self.trainable_parameter_names:
                trainable += [name]
            else:
                fixed += [name]

        logger.info(f"Trainable Parameters:\n\t{trainable}")
        logger.info(f"Fixed Parameters:\n\t{fixed}")

    def log_param_stats(
        self,
        section: str = "params",
        metrics: List[str] = ["min", "max", "norm"],
        log_only_trainable: bool = False,
    ) -> None:
        """
        Log statistics of model parameters.

        Args:
            section (str, optional): Section name for logging. Defaults to "params".
            metrics (List[str], optional): List of metrics to log. Defaults to ["min", "max", "norm"].
            log_only_trainable (bool, optional): Whether to log only trainable parameters. Defaults to False.
        """
        for name, param in self.named_parameters():
            if log_only_trainable and not param.requires_grad:
                continue

            if len(param.data.size()):
                for metric in metrics:
                    self.log(
                        f"{section}/{name}_{metric}",
                        getattr(param.data, metric)(),
                    )
            else:
                self.log(f"{section}/{name}", param.data)

    def get_classifier_dataframe(
        self, layer_name: str = defaults.classifier_name
    ) -> pd.DataFrame:
        """
        Generate a DataFrame containing classifier responses and associated metadata.

        Args:
            layer_name (str, optional): Name of the classifier layer. Defaults to `defaults.classifier_name`.

        Returns:
            pd.DataFrame: DataFrame containing classifier responses and metadata.
        """
        if hasattr(self, "responses"):
            response = self.responses[layer_name]
        else:
            logger.warning("No responses stored!")
            return pd.DataFrame()

        if isinstance(self.label_indices, list):
            label_indices = torch.cat(self.label_indices)
            guess_indices = torch.cat(self.guess_indices)
            image_indices = torch.cat(self.image_indices)
        else:
            label_indices = self.label_indices
            guess_indices = self.guess_indices
            image_indices = self.image_indices

        # cast to cpu and numpy
        valid_data_length = min(len(response), len(label_indices))
        response = response[-valid_data_length:].cpu().float().numpy()
        label_indices = label_indices[-valid_data_length:].cpu().numpy()
        guess_indices = guess_indices[-valid_data_length:].cpu().numpy()
        image_indices = image_indices[-valid_data_length:].cpu().numpy()

        n_samples, n_timesteps, n_classes = response.shape
        sample_indices, times_indices, class_indices = np.meshgrid(
            np.arange(n_samples),
            np.arange(n_timesteps),
            np.arange(n_classes),
            indexing="ij",
        )

        label_sets = np.array(["".join(row.astype(str)) for row in label_indices])
        df = pd.DataFrame(
            {
                "sample_index": sample_indices.ravel(),
                "times_index": times_indices.ravel(),
                "class_index": class_indices.ravel(),
                "response": response.ravel(),
                "label_index": label_indices.ravel().repeat(n_classes),
                "guess_index": guess_indices.ravel().repeat(n_classes),
                "image_index": image_indices.ravel().repeat(n_classes),
                "label_set": label_sets.repeat(n_classes * n_timesteps),
            }
        )
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

    def _concatenate_responses(self) -> None:
        """
        Concatenate stored responses and results, using the int value of the model's attribute 'store_responses' to limit the maximum number of stored responses.
        """
        # Set the maximum number of responses to store (due to memory constraints)
        key = next(iter(self.responses))
        batch_size = self.responses[key][0].shape[0]

        if int(self.store_responses) > 1:
            max_index = int(self.store_responses) // batch_size
            max_index = max(1, max_index)
        else:
            max_index = len(self.responses[key])

        result_attributes = [
            "guess_indices",
            "label_indices",
            "image_indices",
            "times_indices",
        ]

        # Store task results
        for attr_name in result_attributes:
            attribute = getattr(self, attr_name)
            attribute = attribute if isinstance(attribute, list) else [attribute]
            for i, a in enumerate(attribute):
                if a is not None:
                    attribute[i] = a.cpu()

            attribute = self._concatenate_tensors(attribute[-max_index:], dim=0)
            setattr(self, attr_name, attribute)

        # Store task responses
        for layer_name, response in self.responses.items():
            response = response if isinstance(response, list) else [response]
            for i, r in enumerate(response):
                if r is not None:
                    response[i] = r.cpu()

            response = self._concatenate_tensors(response[-max_index:], dim=0)
            self.responses[layer_name] = response

    def _concatenate_tensors(
        self, tensors: List[Optional[torch.Tensor]], dim: int = 1
    ) -> Optional[torch.Tensor]:
        """
        Concatenate a list of tensors along a given dimension.

        Args:
            tensors (List[Optional[torch.Tensor]]): List of tensors to concatenate.
            dim (int, optional): Dimension along which to concatenate. Defaults to 1.

        Returns:
            Optional[torch.Tensor]: Concatenated tensor or None if input is None.
        """
        if tensors is None:
            return None

        # determine common tensor shape
        shape = None
        device = self.device
        for tensor in tensors:
            if tensor is not None:
                shape = tensor.shape
                device = tensor.device
                break

        if shape is None:
            logger.warning("No tensors to concatenate!")
            return None

        for i, tensor in enumerate(tensors):
            if tensor is None:
                tensors[i] = torch.zeros(shape, device=device)
            elif tensor.device != device:
                tensors[i] = tensor.to(device)

        concatenated_tensor = torch.cat(tensors, dim=dim)
        return concatenated_tensor

    def _unsqueeze_tensor(
        self, tensor: Optional[torch.Tensor], dim: int = 1
    ) -> Optional[torch.Tensor]:
        """
        Add a singleton dimension to a tensor.

        Args:
            tensor (Optional[torch.Tensor]): Input tensor.
            dim (int, optional): Dimension to add. Defaults to 1.

        Returns:
            Optional[torch.Tensor]: Tensor with added dimension or None if input is None.
        """
        if tensor is not None:
            tensor = tensor.unsqueeze(dim)
        return tensor

    def _check_for_nans(
        self,
        response_dict: Optional[Dict[str, torch.Tensor]] = None,
        raise_error: bool = False,
    ) -> Optional[None]:
        """
        Check for NaN values in the model responses.

        Args:
            response_dict (Optional[Dict[str, torch.Tensor]], optional): Dictionary of responses to check. Defaults to None.
            raise_error (bool, optional): Whether to raise an error if NaNs are detected. Defaults to False.

        Returns:
            Optional[None]: None if no NaNs are detected.
        """
        if response_dict is None:
            response_dict = self.responses
        for layer in response_dict.keys():
            signal = response_dict[layer]
            if isinstance(signal, list):
                signal = signal[-1]
            if signal is not None and torch.isnan(signal).any():
                logger.warning(f"NaN detected in {layer} responses: ")
                logger.warning(
                    f"\t {torch.isnan(signal).sum().item()} / {signal.numel()}"
                )
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        if len(param.data.size()):
                            logger.info(f"\t{name}_min: {param.data.min().item()}")
                            logger.info(f"\t{name}_max: {param.data.max().item()}")
                            logger.info(f"\t{name}_norm: {param.data.norm().item()}")
                        else:
                            logger.info(f"\t{name}: {param.data}")
                if raise_error:
                    raise ValueError("NaN detected in model responses")
        return None

    def _clear_gpu_memory(self) -> None:
        """
        Clear GPU memory by emptying the cache and synchronizing.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return None

    def safely_named_parameters(self, module=None, replace={".": "_"}):
        if module is None:
            module = self

        if isinstance(module, nn.Sequential):
            generator = (
                (f"Sequential.{child_name}.{param_name}", param)
                for child_name, child in module.named_children()
                if hasattr(child, "named_parameters")
                for param_name, param in child.named_parameters()
            )

        elif hasattr(module, "named_parameters"):
            generator = module.named_parameters()

        else:
            logger.warning(
                f"module {module} has no named parameters to safely rename!"
            )
            generator = iter([])

        for name, param in generator:
            for k, v in replace.items():
                name = name.replace(k, v)
            yield name, param

    def get_safely_named_parameters_dict(self, module=None, replace={".": "_"}):
        if module is None:
            module = self
        if not hasattr(module, "safely_named_parameters_dict"):
            setattr(
                module,
                "safely_named_parameters_dict",
                {
                    k: v
                    for k, v in self.safely_named_parameters(
                        module=module, replace=replace
                    )
                },
            )
        return module.safely_named_parameters_dict


class LightningBase(UtilityBase, pl.LightningModule):
    """
    A base class for PyTorch Lightning models, extending UtilityBase.
    Provides additional functionality for training, validation, and testing.
    """

    @alias_kwargs(
        trc="t_recurrence",
        tff="t_feedforward",
        rctype="recurrence_type",
        lr="learning_rate",
        solver="dynamics_solver",
        taugrad="train_tau",
        lossrt="loss_reaction_time",
    )
    def __init__(
        self,
        input_dims: Tuple[int] = (20, 3, 224, 224),
        retain_graph: bool = defaults.retain_graph,
        store_responses: int = defaults.store_train_responses,
        criterion_params: List[Tuple[str, Dict[str, Any]]] = [
            (loss, defaults.loss_configs[loss]) for loss in defaults.loss
        ],
        loss_reaction_time: float = defaults.loss_reaction_time,
        n_timesteps: int = 1,
        dt: float = defaults.dt,
        tau: float = defaults.tau,
        t_feedforward: float = defaults.t_feedforward,
        t_recurrence: float = defaults.t_recurrence,
        classifier_name: str = defaults.classifier_name,
        store_responses_on_cpu: bool = defaults.store_responses_on_cpu,
        non_label_index: int = defaults.non_label_index,
        optimizer: str = defaults.optimizer,
        optimizer_kwargs: Dict[str, Any] = defaults.optimizer_kwargs,
        optimizer_configs: Dict[str, Dict[str, Any]] = defaults.optimizer_configs,
        learning_rate: float = defaults.learning_rate,
        recurrence_lr_factor: float = defaults.recurrence_lr_factor,
        scheduler: str = defaults.scheduler,
        scheduler_kwargs: Dict[str, Any] = defaults.scheduler_kwargs,
        scheduler_configs: Dict[str, Dict[str, Any]] = defaults.scheduler_configs,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Store the arguments as attributes
        for name, value in kwargs.items():
            if name != "self":
                setattr(self, name, value)
        self.retain_graph = retain_graph
        self.criterion_params = criterion_params
        self.store_responses = store_responses
        self.learning_rate = float(learning_rate)
        self.optimizer = optimizer
        self.dt = float(dt)
        self.tau = float(tau)
        self.t_feedforward = float(t_feedforward)
        self.t_recurrence = float(t_recurrence)
        self.loss_reaction_time = float(loss_reaction_time)
        self.classifier_name = classifier_name
        self.store_responses_on_cpu = store_responses_on_cpu
        self.non_label_index = non_label_index
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_configs = optimizer_configs
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs
        self.scheduler_configs = scheduler_configs
        self.recurrence_lr_factor = recurrence_lr_factor

        # Process the input dims and timesteps
        x = torch.randn(1, *input_dims, device=self.device)
        x = self._adjust_input_dimensions(x)
        _, data_timesteps, self.n_channels, self.dim_y, self.dim_x = x.shape

        if data_timesteps == 1:
            self.n_timesteps = n_timesteps
        elif n_timesteps == 1:
            self.n_timesteps = data_timesteps
        else:
            self.n_timesteps = max(n_timesteps, data_timesteps)
            logger.warning(
                f"The model is initialized with {n_timesteps} timesteps. "
                f"The provided data dimensions have {data_timesteps} timesteps. "
                f"Choosing the larger number of timesteps!"
            )
        self.input_dims = (self.n_timesteps, self.n_channels, self.dim_y, self.dim_x)

        self.save_hyperparameters()  # redundant with logger?

    def setup(self, stage: Optional[str]) -> None:
        """
        Set up the model for training or evaluation.

        Args:
            stage (Optional[str]): Stage of the setup process (e.g., "fit", "test").
        """
        if stage == "fit":
            self._init_parameters()

        self.n_residual_timesteps = self._determine_residual_timesteps(
            device=self.device
        )
        self.reset()
        self._init_loss(self.criterion_params)

    def _define_architecture(self) -> None:
        raise NotImplementedError("Define the model architecture!")

    def _init_parameters(self) -> None:
        for module in self.children():
            if hasattr(module, "_init_parameters"):
                module._init_parameters()

    def verify_initialization(self) -> None:
        pass

    def _forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        feedforward_only: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the model for a single time step. This is a general formulation of the time step operation that uses the model's attributes layer_names and layer_operations to handle the execution of the operations in each layer when they are named according to the following convention: {operation}_{layer_name}, or {operation} if the operation is not layer-specific.
        The main layer computation (e.g. a convolution) can just named {layer_name}.
        Reserved layer operations that are available are 'record' and 'delay' (given that the layer has a set_hidden_state and get_hidden_state method).

        Args:
            x (torch.Tensor): Input tensor for time step t.
            t (Optional[torch.Tensor], optional): Time step index. Defaults to None.
            feedforward_only (bool, optional): Whether to perform only feedforward operations. Defaults to False.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Model output and dictionary of responses.
        """
        batch_size, n_channels, y_dim, x_dim = x.shape

        if hasattr(self, "input_adaption"):
            x = self.input_adaption(x)

        responses = {}
        if not hasattr(self, "layer_operations"):
            # define default operations order within layer
            self.layer_operations = [
                "layer",  # apply (recurrent) convolutional layer
                "addext",  # add external input
                "addskip",  # add skip connection
                "addfeedback",  # add feedback connection
                "tstep",  # apply dynamical systems ode solver step
                "nonlin",  # apply nonlinearity
                "supralin",  # apply supralinearity
                "record",  # record activations in responses dict
                "delay",  # set and get delayed activations for next layer
                "pool",  # apply pooling
                "norm",  # apply normalization
            ]

        for layer_name in self.layer_names:

            layer = getattr(self, layer_name)

            for operation in self.layer_operations:

                if feedforward_only and operation in [
                    "addskip",
                    "addext",
                    "addfeedback",
                ]:
                    continue

                module_name = f"{operation}_{layer_name}"

                if operation == "layer":
                    x = layer(x)

                elif operation == "record":
                    responses[layer_name] = x

                elif operation == "delay" and hasattr(layer, "set_hidden_state"):
                    layer.set_hidden_state(x)
                    x = layer.get_hidden_state(0)

                elif operation == "tstep" and hasattr(self, module_name):
                    module = getattr(self, module_name)
                    h = layer.get_hidden_state(-1)
                    x = module(x, h)

                # apply layer operations (if defined)
                elif hasattr(self, module_name) and x is not None:
                    module = getattr(self, module_name)
                    x = module(x)

                elif hasattr(self, operation) and x is not None:
                    module = getattr(self, operation)
                    x = module(x)

                else:
                    pass

        if x is None:
            x = torch.zeros((batch_size, self.n_classes), device=self.device)
        else:
            classifier = getattr(self, self.classifier_name)
            x = classifier(x)

        responses[self.classifier_name] = x

        return x, responses

    def forward(
        self,
        x_0: torch.Tensor,
        store_responses: bool = False,
        feedforward_only: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x_0 (torch.Tensor): Input tensor.
            store_responses (bool, optional): Whether to store responses. Defaults to False.
            feedforward_only (bool, optional): Whether to perform only feedforward operations. Defaults to False.

        Returns:
            torch.Tensor: Model outputs.
        """
        store_responses = store_responses if store_responses else self.store_responses

        x_0 = self._adjust_input_dimensions(x_0)
        batch_size, n_timesteps, dim_channels, dim_y, dim_x = x_0.shape

        outputs = torch.zeros(
            (batch_size, n_timesteps, self.n_classes), device=x_0.device
        )

        if hasattr(self, "reset"):
            self.reset()

        # Forward the model over all timesteps
        with on_same_device(x_0=x_0, **self.get_safely_named_parameters_dict()):
            for t in torch.arange(n_timesteps, device=x_0.device):
                x = x_0[:, t, ...]

                x, hidden_state = self._forward(
                    x, t, feedforward_only=feedforward_only
                )

                if x is not None:
                    outputs[:, t, :] = x

                if store_responses:
                    self._update_responses(
                        hidden_state, store_n_responses=int(store_responses), t=t
                    )

        del x, hidden_state
        return outputs

    def _init_responses(
        self,
        response_dict: Dict[str, torch.Tensor] = {},
        store_n_responses: Optional[int] = None,
    ) -> None:
        logger.info("Initializing responses")

        if store_n_responses is None:
            store_n_responses = int(self.store_responses)

        if self.store_responses_on_cpu:
            device = "cpu"
        else:
            device = self.device

        self.responses = {}
        self.reset()

        if not response_dict or any(v is None for v in response_dict.values()):
            random_input = torch.randn(
                (1, self.n_channels, self.dim_y, self.dim_x), device=self.device
            )
            while any(v is None for v in response_dict.values()):
                _, response_dict = self._forward(random_input, feedforward_only=True)

        if hasattr(self, "n_residual_timesteps"):
            n_timesteps = self.n_timesteps + self.n_residual_timesteps
        else:
            n_timesteps = self.n_timesteps

        # n_timesteps = 40  ## hack!!
        for layer_name in response_dict.keys():
            self.responses[layer_name] = torch.zeros(
                (
                    store_n_responses,
                    n_timesteps,
                    *response_dict[layer_name].shape[1:],
                ),
                device=device,
            )
        self.reset()
        return None

    def _update_responses(
        self,
        response_dict: Dict[str, torch.Tensor],
        store_n_responses: Optional[int] = None,
        t: Optional[int] = None,
    ) -> None:
        if not hasattr(self, "responses"):
            self._init_responses(response_dict, store_n_responses=store_n_responses)

        for layer_name, response in response_dict.items():
            if response is None:
                continue
            elif response.device != self.responses[layer_name].device:
                response = response.to(self.responses[layer_name].device)
            else:
                pass

            batch_size = response.shape[0]
            layer_response_size = self.responses[layer_name].shape[0]

            if t is None:
                if layer_response_size > batch_size:
                    # shift the responses to the left (mimic deque behavior)
                    response_history = self.responses[layer_name][batch_size:, ...]
                    self.responses[layer_name][:batch_size, ...] = response_history
                    self.responses[layer_name][-batch_size:, ...] = response
                else:
                    self.responses[layer_name][:, ...] = response[
                        -layer_response_size:
                    ]
            else:
                if layer_response_size > batch_size:
                    # shift the responses to the left (mimic deque behavior)
                    response_history = self.responses[layer_name][batch_size:, t, ...]
                    self.responses[layer_name][:-batch_size, t, ...] = response_history
                    self.responses[layer_name][-batch_size:, t, ...] = response
                else:
                    self.responses[layer_name][:, t, ...] = response[
                        -layer_response_size:
                    ]
        return None

    def predictor(self, outputs: torch.Tensor) -> torch.Tensor:
        return torch.argmax(outputs, dim=-1)

    def _expand_timesteps(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, label_indices, *extra = batch
        inputs = self._adjust_input_dimensions(inputs)
        label_indices = self._adjust_label_dimensions(label_indices)

        if inputs.size(1) == 1 and self.n_timesteps > 1:
            # input data is not yet extended
            inputs = inputs.expand(-1, self.n_timesteps, -1, -1, -1)
            label_indices = label_indices.expand(-1, self.n_timesteps)

        return (inputs, label_indices, *extra)

    def _extend_residual_timesteps(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, label_indices, *extra = batch

        inputs = self._adjust_input_dimensions(inputs)
        label_indices = self._adjust_label_dimensions(label_indices)

        if self.n_residual_timesteps > 0:
            # add 0s at the end as inputs for residual timesteps
            new_shape = (
                inputs.size(0),
                self.n_timesteps + self.n_residual_timesteps,
                *inputs.shape[2:],
            )
            new_inputs = torch.zeros(
                new_shape, device=inputs.device, dtype=inputs.dtype
            )
            new_inputs[:, : self.n_timesteps, ...] = inputs

            # add voidid buffer labels at the beginning for residual timesteps

            new_shape = (inputs.size(0), self.n_timesteps + self.n_residual_timesteps)
            new_label_indices = torch.full(
                new_shape,
                self.non_label_index,
                device=label_indices.device,
                dtype=label_indices.dtype,
            )
            new_label_indices[:, -self.n_timesteps :] = label_indices

            batch = (new_inputs, new_label_indices, *extra)

        else:

            batch = (inputs, label_indices, *extra)

        return batch

    def model_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        store_responses: bool = False,
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        batch = self._expand_timesteps(batch)
        inputs, label_indices, *paths = self._extend_residual_timesteps(batch)

        # forward
        # (you may override the model's store_response setting here)
        outputs = self.forward(inputs, store_responses=store_responses)

        # calculate loss
        loss = self.compute_loss(
            outputs,
            responses=self.responses if hasattr(self, "responses") else None,
            label_indices=label_indices,
            reaction_time=self.loss_reaction_time,
        )

        # calculate accuracy
        guess_indices = self.predictor(outputs)
        accuracy = self.calc_accuracy(label_indices, guess_indices)

        del outputs
        return loss, accuracy, guess_indices

    def _init_loss(self, criterion_params: List[Tuple[str, Dict[str, Any]]]) -> None:
        if hasattr(self, "criterion"):
            if isinstance(self.criterion, list) and (
                isinstance(self.criterion[0], nn.Module)
                or isinstance(self.criterion[0], tuple)
            ):
                # already initialized
                return None
            elif self.criterion is None:
                pass
            else:
                raise ValueError(
                    "self.criterion should be a list of loss functions or tuples of loss functions and weights"
                )

        self.criterion = []

        if not isinstance(criterion_params, list):
            criterion_params = [criterion_params]

        for criterion_name, criterion_config in criterion_params:
            if "weight" in criterion_config.keys():
                criterion_weight = criterion_config.pop("weight")
            else:
                criterion_weight = 1

            logger.info(
                f"Criterion: {criterion_name} with weight: {criterion_weight} and config: {criterion_config}"
            )

            if hasattr(losses, criterion_name):
                criterion_fn = getattr(losses, criterion_name)(**criterion_config)
                self.criterion += [(criterion_fn, criterion_weight)]
            else:
                raise ValueError(f"Invalid loss function: {criterion_name}")

        return None

    def compute_loss(
        self,
        outputs: torch.Tensor,
        label_indices: torch.Tensor,
        responses: Optional[Dict[str, torch.Tensor]] = None,
        reaction_time: float = 0,
    ) -> torch.Tensor:
        self._init_loss(self.criterion_params)

        *_, n_classes = outputs.shape

        if reaction_time:
            reaction_timesteps = self.n_residual_timesteps + int(
                reaction_time / self.dt
            )
            outputs = (
                outputs[:, reaction_timesteps:, :].contiguous().view(-1, n_classes)
            )
            label_indices = label_indices[:, reaction_timesteps:].contiguous().view(-1)
        else:
            outputs = outputs.view(-1, n_classes)
            label_indices = label_indices.view(-1)

        loss_values = torch.zeros(len(self.criterion), device=outputs.device)
        for i, criterion_fn in enumerate(self.criterion):
            if isinstance(criterion_fn, tuple):
                criterion_fn, weight = criterion_fn
            else:
                weight = 1

            loss_value = weight * criterion_fn((outputs, responses), label_indices)

            if torch.isnan(loss_value):
                logger.warning(f"Loss Value: {criterion_fn}, {loss_value}")
                logger.warning(f"Output contains NaNs: {torch.isnan(outputs).any()}")
                logger.warning("Warning: Loss value is NaN")

            loss_values[i] = loss_value

        loss = loss_values.sum()
        return loss

    def calc_accuracy(
        self,
        label_indices: torch.Tensor,
        guess_indices: torch.Tensor,
    ) -> float:
        mask = torch.where(label_indices >= 0)
        accuracy = (guess_indices[mask] == label_indices[mask]).float().mean().item()
        return accuracy

    def backward(self, loss: torch.Tensor) -> None:
        if not self.retain_graph:
            try:
                loss.backward()
            except Exception as e:
                logger.error(e)
                logger.warning("Setting `retain_graph` to True!")
                self.retain_graph = True

        if self.retain_graph:
            loss.backward(retain_graph=True)

    def _shared_eval_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        store_responses: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        batch = self._expand_timesteps(batch)
        inputs, label_indices, *paths = self._extend_residual_timesteps(batch)
        loss, accuracy, guess_indices = self.model_step(
            batch, batch_idx, store_responses
        )

        batch_size, n_timesteps = guess_indices.shape

        self.guess_indices.append(guess_indices)
        self.label_indices.append(label_indices)

        if paths and (isinstance(paths[0], str) or isinstance(paths[0], Path)):
            image_indices = torch.tensor(
                [path_to_index(path) for path in paths], device=inputs.device
            )
        else:
            image_indices = (
                torch.arange(batch_size, device=inputs.device) + batch_idx * batch_size
            )

        image_indices = image_indices.unsqueeze(1).expand(batch_size, n_timesteps)
        self.image_indices.append(image_indices)

        times_indices = torch.arange(n_timesteps, device=inputs.device)
        times_indices = times_indices.unsqueeze(0).expand(batch_size, n_timesteps)
        self.times_indices.append(times_indices)
        return loss, accuracy

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Batch of input data and labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Training loss.
        """
        batch_size = batch[0].size(0)
        loss, accuracy, *_ = self.model_step(batch, batch_idx, store_responses=False)

        metrics = {"train_loss": loss, "train_accuracy": accuracy}
        self.log_dict(metrics, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, float]:
        """
        Perform a single validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Batch of input data and labels.
            batch_idx (int): Index of the batch.

        Returns:
            Tuple[torch.Tensor, float]: Validation loss and accuracy.
        """
        batch_size = batch[0].size(0)
        loss, accuracy = self._shared_eval_step(
            batch, batch_idx, store_responses=self.store_responses
        )

        metrics = {"val_loss": loss, "val_accuracy": accuracy}
        self.log_dict(metrics, prog_bar=True, batch_size=batch_size)

        self._clear_gpu_memory()
        return loss, accuracy

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, float]:
        """
        Perform a single test step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Batch of input data and labels.
            batch_idx (int): Index of the batch.

        Returns:
            Tuple[torch.Tensor, float]: Test loss and accuracy.
        """
        batch_size = batch[0].size(0)
        loss, accuracy = self._shared_eval_step(
            batch, batch_idx, store_responses=self.store_responses
        )

        metrics = {"test_loss": loss, "test_accuracy": accuracy}
        self.log_dict(metrics, prog_bar=True, on_step=True, batch_size=batch_size)

        return loss, accuracy

    def optimizer_step(self, *args: Any, **kwargs: Any) -> None:
        super().optimizer_step(*args, **kwargs)

    def on_train_start(self) -> None:
        pass

    def on_train_epoch_start(self) -> None:
        pass

    def on_train_batch_end(
        self, outputs: Any, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        self.log_param_stats()

    def on_test_start(self) -> None:
        self.guess_indices = []
        self.label_indices = []
        self.image_indices = []
        self.times_indices = []

    def on_validation_start(self) -> None:
        self.on_test_start()

    def on_test_epoch_end(self) -> None:
        pass

    def on_test_batch_end(
        self,
        outputs: Optional[Any] = None,
        batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        batch_idx: int = -1,
    ) -> None:
        pass

    def on_validation_end(
        self,
        outputs: Optional[Any] = None,
        batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        batch_idx: int = -1,
    ) -> None:
        pass

    def configure_optimizers(
        self, recurrence_lr_factor: float = None
    ) -> Dict[str, Any]:
        """
        Configure the optimizers and learning rate schedulers for the model.

        Args:
            recurrence_lr_factor (float, optional): Factor to reduce learning rate for recurrent weights. Defaults to `self.recurrence_lr_factor`.

        Returns:
            Dict[str, Any]: Dictionary containing optimizer and scheduler configurations.
        """
        if recurrence_lr_factor is None:
            recurrence_lr_factor = self.recurrence_lr_factor

        # Retrieve the base learning rate
        lr = getattr(self, "lr", getattr(self, "learning_rate", None))

        # Adjust the learning rate for recurrent weights
        lr_recurrence = lr * recurrence_lr_factor

        # Initialize lists to hold recurrent and non-recurrent parameters
        recurrence_params, non_recurrence_params = [], []

        # Log trainable parameter names for debugging
        self.print_trainable_parameter_names()

        # Retrieve the optimizer class from PyTorch
        if isinstance(self.optimizer, str):
            if hasattr(torch.optim, self.optimizer):
                optimizer_class = getattr(torch.optim, self.optimizer)
            else:
                raise ValueError(f"Unknown optimizer: {self.optimizer}")

            # Split parameters into recurrent and non-recurrent groups
            for name, param in self.named_trainable_parameters():
                if "recurrence" in name:  # Identify recurrent parameters by name
                    recurrence_params.append(param)
                else:
                    non_recurrence_params.append(param)

            # Prepare optimizer parameter groups
            param_groups = [{"params": non_recurrence_params}]
            if recurrence_params:
                param_groups.append({"params": recurrence_params, "lr": lr_recurrence})

            # Instantiate the optimizer with the parameter groups
            optimizer = optimizer_class(param_groups, lr=lr, **self.optimizer_kwargs)
        else:
            # Use the provided optimizer instance if passed directly
            optimizer = self.optimizer

        # Configure the learning rate scheduler
        if hasattr(self, "lr_scheduler") and hasattr(
            torch.optim.lr_scheduler, self.lr_scheduler
        ):
            # Use the specified scheduler and its arguments
            if not hasattr(self, "lr_scheduler_kwargs"):
                self.lr_scheduler_kwargs = {}
        else:
            # Default scheduler settings
            self.lr_scheduler = self.scheduler
            self.lr_scheduler_kwargs = self.scheduler_kwargs

        # Instantiate the learning rate scheduler
        lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler)(
            optimizer, **self.lr_scheduler_kwargs
        )

        # Prepare the scheduler configuration
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            **self.scheduler_configs,
        }

        # Return the optimizer and scheduler configuration
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
            **self.optimizer_configs,
        }

    def log_table(
        self,
        key: str,
        columns: Optional[List[str]] = None,
        data: Optional[List[List[Any]]] = None,
        dataframe: Optional[pd.DataFrame] = None,
        step: Optional[int] = None,
    ) -> None:
        if self.logger:
            self.logger.log_table(key=key, dataframe=dataframe, step=step)

    def log_figure(
        self, fig: plt.Figure, key: str, step: Optional[int] = None
    ) -> None:
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plt.close()
        self.log({f"{key}", wandb.Image(buffer, caption=key)}, step=step)


if __name__ == "__main__":
    input_shape = (1, 28, 28)

    model = LightningBase(
        input_dims=input_shape, n_timesteps=8, n_classes=10, cumulative_readout=True
    )

    random_input = torch.randn(2, *input_shape)

    outputs = model(random_input)

    print(f"Random Input ({random_input.shape})")
    print(f"Model Output Timesteps: {len(outputs)}")
    for t, output in enumerate(outputs):
        print(f"\tt {t}: ({output.shape}): \t{output}")
