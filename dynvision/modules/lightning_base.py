import inspect
import io
from collections import defaultdict
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb

from dynvision import losses
from dynvision.utils.utils import (
    alias_kwargs,
    path_to_index,
    load_config,
)
from dynvision.visualization.plot_classifier_responses import (
    plot_classifier_responses,
)
from dynvision.project_paths import project_paths

__all__ = ["UtilityBase", "LightningBase"]

defaults = SimpleNamespace(
    **load_config(project_paths.scripts.configs / "config_defaults.yaml")
)


class UtilityBase(nn.Module):

    def _define_architecture(self):
        raise NotImplementedError("Define the model architecture!")

    def reset(self):
        # reset hidden states here
        pass

    def _adjust_input_dimensions(self, x):
        """
        Input dimensions are expected to be in one of the format:
            - dim_y, dim_x
            - batch_size, dim_y, dim_x
            - batch_size, dim_channels, dim_y, dim_x
            - batch_size, n_timesteps, dim_channels, dim_y, dim_x
        Return the input dimensions in the format:
            - batch_size, n_timesteps, dim_channels, dim_y, dim_x
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 3:
            x = x.unsqueeze(1).unsqueeze(1)
        elif len(x.shape) == 4:
            x = x.unsqueeze(1)
        elif len(x.shape) == 5:
            pass
        else:
            raise ValueError(f"Invalid input shape: {x.shape}")
        return x

    def _adjust_label_dimensions(self, x):
        """
        Input dimensions are expected to be in one of the format:
            - batch_size
            - batch_size, n_timesteps
        Return the input dimensions in the format:
            - batch_size, n_timesteps
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        elif len(x.shape) == 2:
            pass
        else:
            raise ValueError(f"Invalid label shape: {x.shape}")
        return x

    def _expand_input_channels(
        self, x: torch.Tensor, n_target_channels: int = 3
    ) -> torch.Tensor:
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
        self, input_shape, kernel_size, stride, padding=0
    ):
        return tuple(
            (dim - kernel_size + 2 * padding) // stride + 1 for dim in input_shape
        )

    def _determine_residual_timesteps(self):
        # dtype = next(self.parameters()).dtype
        random_input = torch.randn(
            (1, self.n_channels, self.dim_y, self.dim_x), device=self.device
        )
        null_input = torch.zeros_like(random_input, device=self.device)

        # null_output, _ = self._forward(null_input)

        if hasattr(self, "reset"):
            self.reset()

        x = None
        t = -1
        is_empty_output = lambda x: (x is None) or (
            torch.all(x.eq(0))
            or (torch.isnan(x).all())
            # or (torch.allclose(x, null_output, atol=1e-6))
            or x.grad_fn is None
        )

        while is_empty_output(x):
            x, _ = self._forward(random_input, feedforward_only=True)
            t += 1
            if t > 100:
                raise ValueError("Unable to determine residual timesteps (>100)!")

        if hasattr(self, "reset"):
            self.reset()

        return t

    def _add_missing_parameters_to_state_dict(self, state_dict):
        missing_parameter_names = []
        for key in self.state_dict().keys():
            if key not in state_dict.keys():
                missing_parameter_names.append(key)
                state_dict[key] = self.state_dict()[key]
        if missing_parameter_names:
            print(
                f"Adding missing parameters to loaded state dict: {missing_parameter_names}"
            )
        return state_dict

    def _remove_unexpected_parameters_from_state_dict(self, state_dict):
        unexpected_parameter_names = []
        for key in state_dict.keys():
            if key not in self.state_dict().keys():
                unexpected_parameter_names.append(key)
        if unexpected_parameter_names:
            print(
                f"Removing unexpected parameters from loaded state dict: {unexpected_parameter_names}"
            )
            for key in unexpected_parameter_names:
                del state_dict[key]
        return state_dict

    def load_pretrained_state_dict(self, check_mismatch_layer=[], strict=True):

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

    def load_state_dict(self, state_dict, **kwargs):
        state_dict = self._add_missing_parameters_to_state_dict(state_dict)
        state_dict = self._remove_unexpected_parameters_from_state_dict(state_dict)
        super(UtilityBase, self).load_state_dict(state_dict, **kwargs)

    def trainable_parameters(self):
        if hasattr(self, "trainable_parameter_names"):
            for name, param in self.named_parameters():
                if name in self.trainable_parameter_names:
                    yield param
        else:
            yield from self.parameters()

    def named_trainable_parameters(self):
        if hasattr(self, "trainable_parameter_names"):
            for name, param in self.named_parameters():
                if name in self.trainable_parameter_names:
                    yield name, param
        else:
            yield from self.named_parameters()

    def print_trainable_parameter_names(self):
        trainable, fixed = [], []
        for name, param in self.named_parameters():
            if (
                hasattr(self, "trainable_parameter_names")
                and name in self.trainable_parameter_names
                and param.requires_grad
            ):
                trainable += [name]
            elif (
                not hasattr(self, "trainable_parameter_names") and param.requires_grad
            ):
                trainable += [name]
            else:
                fixed += [name]

        print(f"Trainable Parameters:\n\t{trainable}")
        print(f"Fixed Parameters:\n\t{fixed}")

    def log_param_stats(self, section="params", metrics=["min", "max", "norm"]):
        for name, param in self.named_trainable_parameters():
            if not param.requires_grad:
                continue

            if len(param.data.size()):
                for metric in metrics:
                    self.log(
                        f"{section}/{name}_{metric}", getattr(param.data, metric)()
                    )
            else:
                self.log(f"{section}/{name}", param.data)

    def get_classifier_dataframe(self, layer_name=defaults.classifier_name):
        if hasattr(self, "responses"):
            response = self.responses[layer_name]
        else:
            print("No responses stored!")
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

    def _concatenate_responses(self):
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

    def _concatenate_tensors(self, tensors: list, dim: int = 1):
        """
        Concatenate a list of tensors along a given dimension.
        Elements that are None are replaced with zeros.
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
            print("No tensors to concatenate!")
            return None

        for i, tensor in enumerate(tensors):
            if tensor is None:
                tensors[i] = torch.zeros(shape, device=device)
            elif tensor.device != device:
                tensors[i] = tensor.to(device)

        concatenated_tensor = torch.cat(tensors, dim=dim)
        return concatenated_tensor

    def _unsqueeze_tensor(self, tensor, dim=1):
        if tensor is not None:
            tensor = tensor.unsqueeze(dim)
        return tensor

    def _check_for_nans(self, response_dict=None, raise_error=False):
        if response_dict is None:
            response_dict = self.responses
        for layer in response_dict.keys():
            signal = response_dict[layer]
            if isinstance(signal, list):
                signal = signal[-1]
            if signal is not None and torch.isnan(signal).any():
                print(f"NaN detected in {layer} responses: ")
                print("\t", torch.isnan(signal).sum().item(), "/", signal.numel())
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        if len(param.data.size()):
                            print("\t", f"{name}_min", param.data.min().item())
                            print("\t", f"{name}_max", param.data.max().item())
                            print("\t", f"{name}_norm", param.data.norm().item())
                        else:
                            print("\t", f"{name}", param.data)
                if raise_error:
                    raise ValueError("NaN detected in model responses")
        return None

    def _clear_gpu_memory(self):
        # print(
        #     f"GPU memory occupied: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB"
        # )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return None


class LightningBase(UtilityBase, pl.LightningModule):
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
        *args,
        retain_graph: bool = defaults.retain_graph,
        store_responses: int = defaults.store_train_responses,
        criterion_params=[
            (loss, defaults.loss_configs[loss]) for loss in defaults.loss
        ],
        learning_rate: float = defaults.learning_rate,
        optimizer=defaults.optimizer,
        loss_reaction_time: float = defaults.loss_reaction_time,
        dt: float = defaults.dt,
        tau: float = defaults.tau,
        t_feedforward: float = defaults.t_feedforward,
        t_recurrence: float = defaults.t_recurrence,
        **kwargs,
    ):

        super(LightningBase, self).__init__()

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

        # Define the input dims
        if "input_dims" in kwargs:
            x = torch.randn(1, *kwargs["input_dims"], device=self.device)
            x = self._adjust_input_dimensions(x)
            _, self.n_timesteps, self.n_channels, self.dim_y, self.dim_x = x.shape

        # Define model architecture
        self._define_architecture()
        self._init_parameters()
        self._init_loss(criterion_params)

        # Determine residual timesteps
        self.n_residual_timesteps = self._determine_residual_timesteps()

        # Reset hidden states
        self.reset()

        # Log the hyperparameters
        self.save_hyperparameters()

    def _define_architecture(self):
        raise NotImplementedError("Define the model architecture!")

    def _init_parameters(self):
        for module in self.children():
            if hasattr(module, "_init_parameters"):
                module._init_parameters()

    def verify_initialization(self):
        pass

    def _forward(
        self, x: torch.Tensor, t: torch.Tensor = None, feedforward_only: bool = False
    ) -> torch.Tensor:
        batch_size, n_channels, y_dim, x_dim = x.shape

        if hasattr(self, "input_adaption"):
            x = self.input_adaption(x)

        responses = {}
        if not hasattr(self, "layer_operations"):
            # define operations order within layer
            self.layer_operations = [
                "layer",  # apply (recurrent) convolutional layer
                "addskip",  # add skip connection
                "addext",  # add external input
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
                if feedforward_only and operation in ["addskip", "addext"]:
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
                    x = module(x, h=layer.get_hidden_state(-1))

                # apply layer operations (if defined)
                elif hasattr(self, module_name) and x is not None:
                    module = getattr(self, module_name)
                    # signature = inspect.signature(module)
                    # if "t" in signature.parameters and t is not None:
                    #     x = module(x, t=t)
                    # else:
                    x = module(x)

                elif hasattr(self, operation) and x is not None:
                    module = getattr(self, operation)
                    x = module(x)

                else:
                    pass

        # linear classifier
        if x is None:
            x = torch.zeros((batch_size, self.n_classes), device=self.device)
        else:
            classifier = getattr(self, defaults.classifier_name)
            x = classifier(x)

        responses[defaults.classifier_name] = x

        # debugging: check for NaNs in outputs
        # self._check_for_nans(responses)

        return x, responses

    def forward(
        self,
        x_0: torch.Tensor,
        store_responses: bool = False,
        feedforward_only: bool = False,
        *args,
        **kwargs,
    ):
        store_responses = store_responses if store_responses else self.store_responses

        x_0 = self._adjust_input_dimensions(x_0)
        batch_size, n_timesteps, dim_channels, dim_y, dim_x = x_0.shape

        outputs = torch.zeros(
            (batch_size, n_timesteps, self.n_classes), device=x_0.device
        )
        # responses = {}  # defaultdict(list)

        if hasattr(self, "reset"):
            self.reset()

        # Forward the model over all timesteps
        for t in torch.arange(n_timesteps, device=x_0.device):
            x = x_0[:, t, ...]

            x, hidden_state = self._forward(x, t, feedforward_only=feedforward_only)

            # outputs.append(self._unsqueeze_tensor(x, dim=1))
            if x is not None:
                outputs[:, t, :] = x

            if store_responses:
                # ToDo: updating involves shifting values in the response tensors, maybe it is more efficient to store each batch and update only at the end of forward pass? Evaluate!
                self._update_responses(
                    hidden_state, store_n_responses=int(store_responses), t=t
                )

        # # outputs = self._concatenate_tensors(outputs, dim=1)

        # # Store the responses
        # # (either all or only the latest, depending on the store_responses setting)
        # for layer_name, response in responses.items():
        #     # batch_response = self._concatenate_tensors(response, dim=1)

        #     if not store_responses or (1 < int(store_responses) <= batch_size):
        #         self.responses[layer_name] = [batch_response]
        #     else:
        #         self.responses[layer_name].append(batch_response)

        # del responses
        del x, hidden_state
        return outputs

    def _init_responses(self, response_dict={}, store_n_responses=None):
        if store_n_responses is None:
            store_n_responses = int(self.store_responses)

        if defaults.store_responses_on_cpu:
            device = "cpu"
        else:
            device = self.device

        self.responses = {}

        if not response_dict or any(v is None for v in response_dict.values()):
            random_input = torch.randn(
                (1, self.n_channels, self.dim_y, self.dim_x), device=self.device
            )
            while any(v is None for v in response_dict.values()):
                _, response_dict = self._forward(random_input, feedforward_only=True)

        if hasattr(self, "n_residual_timesteps"):
            n_timesteps = self.n_timesteps + self.n_residual_timesteps

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

    def _update_responses(self, response_dict, store_n_responses=None, t=None):
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

    def predictor(self, outputs):
        return torch.argmax(outputs, dim=-1)

    def _expand_residual_timesteps(self, batch):
        inputs, label_indices, *paths = batch

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

            # add -1 buffer labels at the beginning for residual timesteps
            new_shape = (inputs.size(0), self.n_timesteps + self.n_residual_timesteps)
            new_label_indices = torch.full(
                new_shape, -1, device=label_indices.device, dtype=label_indices.dtype
            )
            new_label_indices[:, -self.n_timesteps :] = label_indices

            batch = (new_inputs, new_label_indices, *paths)

        else:

            batch = (inputs, label_indices, *paths)

        return batch

    def model_step(self, batch, batch_idx, store_responses=False):
        inputs, label_indices, *paths = self._expand_residual_timesteps(batch)

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

    def _init_loss(self, criterion_params):
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

            if hasattr(losses, criterion_name):
                criterion_fn = getattr(losses, criterion_name)(**criterion_config)
                self.criterion += [(criterion_fn, criterion_weight)]
            else:
                raise ValueError(f"Invalid loss function: {criterion_name}")

        return None

    def compute_loss(self, outputs, label_indices, responses=None, reaction_time=0):
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
                print("Loss Value:", criterion_fn, loss_value)
                print("Output contains NaNs:", torch.isnan(outputs).any())
                print("Warning: Loss value is NaN")

            loss_values[i] = loss_value

        loss = loss_values.sum()
        return loss

    def calc_accuracy(self, label_indices, guess_indices, ignore_index=-1):
        mask = torch.where(label_indices != ignore_index)
        accuracy = (guess_indices[mask] == label_indices[mask]).float().mean().item()
        return accuracy

    def backward(self, loss):
        if not self.retain_graph:
            try:
                loss.backward()
            except Exception as e:
                print(e)
                print("Setting `retain_graph` to True!")
                self.retain_graph = True

        if self.retain_graph:
            loss.backward(retain_graph=True)

    def _shared_eval_step(self, batch, batch_idx, store_responses=False):
        inputs, label_indices, *paths = self._expand_residual_timesteps(batch)
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

    def training_step(self, batch, batch_idx):
        batch_size = batch[0].size(0)
        loss, accuracy, *_ = self.model_step(batch, batch_idx, store_responses=False)

        metrics = {"train_loss": loss, "train_accuracy": accuracy}
        self.log_dict(metrics, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch[0].size(0)
        loss, accuracy = self._shared_eval_step(
            batch, batch_idx, store_responses=self.store_responses
        )

        metrics = {"val_loss": loss, "val_accuracy": accuracy}
        self.log_dict(metrics, prog_bar=True, batch_size=batch_size)

        self._clear_gpu_memory()
        return loss, accuracy

    def test_step(self, batch, batch_idx):
        batch_size = batch[0].size(0)
        loss, accuracy = self._shared_eval_step(
            batch, batch_idx, store_responses=self.store_responses
        )

        metrics = {"test_loss": loss, "test_accuracy": accuracy}
        self.log_dict(metrics, prog_bar=True, on_step=True, batch_size=batch_size)

        self._clear_gpu_memory()
        return loss, accuracy

    def optimizer_step(self, *args, **kwargs):
        super(LightningBase, self).optimizer_step(*args, **kwargs)
        self._clear_gpu_memory()

    def on_train_start(self):
        pass

    def on_train_epoch_start(self):
        pass

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.log_param_stats()

    def on_test_start(self):
        self.guess_indices = []
        self.label_indices = []
        self.image_indices = []
        self.times_indices = []

    def on_validation_start(self):
        self.on_test_start()

    def on_test_epoch_end(self):
        # self._concatenate_responses()
        pass

    def on_test_batch_end(self, outputs=None, batch=None, batch_idx=-1):
        pass

    def on_validation_end(self, outputs=None, batch=None, batch_idx=-1):
        pass

    def configure_optimizers(self, recurrence_lr_factor=defaults.recurrence_lr_factor):
        lr = getattr(self, "lr", getattr(self, "learning_rate", None))

        # Reduce the learning rate for the recurrence weights
        lr_recurrence = lr * recurrence_lr_factor

        # Split the parameters into recurrent and non-recurrent
        recurrence_params, params = [], []

        if isinstance(self.optimizer, str):
            if hasattr(torch.optim, self.optimizer):
                optimizer_class = getattr(torch.optim, self.optimizer)
            else:
                raise ValueError(f"Unknown optimizer: {self.optimizer}")

            self.print_trainable_parameter_names()

            # reduce learning rate for recurrent weights
            for name, param in self.named_parameters():
                if name in self.trainable_parameter_names:
                    if "recurrence" in name:
                        recurrence_params.append(param)
                    else:
                        params.append(param)

            # create the optimizers
            params_args = [{"params": params}]
            if recurrence_params:
                params_args.append({"params": recurrence_params, "lr": lr_recurrence})

            optimizer = optimizer_class(
                params_args, lr=lr, **defaults.optimizer_kwargs
            )
        else:
            optimizer = self.optimizer

        # create the learning rate scheduler
        if hasattr(self, "lr_scheduler") and hasattr(
            torch.optim.lr_scheduler, self.lr_scheduler
        ):
            if not hasattr(self, "lr_scheduler_kwargs"):
                self.lr_scheduler_kwargs = {}
        else:  # default settings
            self.lr_scheduler = defaults.scheduler
            self.lr_scheduler_kwargs = defaults.scheduler_kwargs

        lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler)(
            optimizer, **self.lr_scheduler_kwargs
        )

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            **defaults.scheduler_configs,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
            **defaults.optimizer_configs,
        }

    def log_table(self, key, columns=None, data=None, dataframe=None, step=None):
        if self.logger:
            self.logger.log_table(key=key, dataframe=dataframe, step=step)

    def log_figure(self, fig, key, step=None):
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

    # print(summary(model, input_shape))

    random_input = torch.randn(2, *input_shape)

    outputs = model(random_input)

    print(f"Random Input ({random_input.shape})")
    print(f"Model Output Timesteps: {len(outputs)}")
    for t, output in enumerate(outputs):
        print(f"\tt {t}: ({output.shape}): \t{output}")

    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.plot(outputs.detach().numpy().squeeze())
    # plt.show()
