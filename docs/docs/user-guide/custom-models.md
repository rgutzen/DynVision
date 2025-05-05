# Creating Custom Models

DynVision is designed to be extensible, allowing you to create custom neural network architectures with biologically plausible properties. This guide will walk you through the process of defining your own models using DynVision's components and base classes.

## Overview

Creating a custom model in DynVision typically involves these steps:

1. Inheriting from the appropriate base class
2. Defining the model architecture
3. Implementing required methods
4. Customizing parameters and behaviors

## Base Classes

DynVision provides several base classes for model creation:

1. **UtilityBase**: Core functionality for neural network models (inherits from `nn.Module`)
2. **LightningBase**: Integration with PyTorch Lightning for training and evaluation (inherits from `UtilityBase` and `pl.LightningModule`)
3. **DyRCNN**: Base class specifically for recurrent convolutional networks (inherits from `LightningBase`)

For most custom models, you'll want to inherit from `LightningBase` or `DyRCNN`.

## Simple Custom Model Example

Let's create a simple 2-layer recurrent model:

```python
import torch
import torch.nn as nn
from dynvision.model_components import RecurrentConnectedConv2d, EulerStep
from dynvision.model_components import LightningBase

class SimpleRCNN(LightningBase):
    def __init__(
        self, 
        n_classes=10, 
        input_dims=(20, 3, 32, 32),
        recurrence_type="self",
        **kwargs
    ):
        super().__init__(
            n_classes=n_classes, 
            input_dims=input_dims,
            recurrence_type=recurrence_type,
            **kwargs
        )
        self._define_architecture()
    
    def _define_architecture(self):
        """Define the model architecture."""
        self.layer_names = ["layer1", "layer2"]
        # Define operations order within layer
        self.layer_operations = [
            "layer",        # Apply (recurrent) convolutional layer
            "tstep",        # Apply dynamical systems ODE solver step
            "nonlin",       # Apply nonlinearity
            "record",       # Record activations in responses dict
            "delay",        # Set and get delayed activations for next layer
            "pool"          # Apply pooling
        ]
        
        # Define activation function
        self.nonlin = nn.ReLU(inplace=False)
        
        # Define layer1
        self.layer1 = RecurrentConnectedConv2d(
            in_channels=self.n_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            dim_y=self.dim_y,
            dim_x=self.dim_x,
            bias=True,
            recurrence_type=self.recurrence_type,
            dt=self.dt,
            tau=self.tau,
            history_length=self.t_feedforward,
            recurrence_delay=self.t_recurrence,
            device=self.device
        )
        self.tau_layer1 = torch.nn.Parameter(
            torch.tensor(self.tau, dtype=float),
            requires_grad=self.train_tau,
        )
        self.tstep_layer1 = EulerStep(dt=self.dt, tau=self.tau_layer1)
        self.pool_layer1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Define layer2
        self.layer2 = RecurrentConnectedConv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            dim_y=self.dim_y // 2,  # After pooling in layer1
            dim_x=self.dim_x // 2,  # After pooling in layer1
            bias=True,
            recurrence_type=self.recurrence_type,
            dt=self.dt,
            tau=self.tau,
            history_length=self.t_feedforward,
            recurrence_delay=self.t_recurrence,
            device=self.device
        )
        self.tau_layer2 = torch.nn.Parameter(
            torch.tensor(self.tau, dtype=float),
            requires_grad=self.train_tau,
        )
        self.tstep_layer2 = EulerStep(dt=self.dt, tau=self.tau_layer2)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, self.n_classes)
        )
    
    def reset(self):
        """Reset the model's internal states."""
        self.layer1.reset()
        self.tstep_layer1.reset()
        self.layer2.reset()
        self.tstep_layer2.reset()
```

This model implements a 2-layer RCNN with recurrent connections and neural dynamics.

## Using the Custom Model

You can now use your custom model just like built-in models:

```python
from my_models import SimpleRCNN

# Create model
model = SimpleRCNN(
    n_classes=10,
    input_dims=(20, 3, 32, 32),
    recurrence_type="full",
    dt=2,
    tau=8,
    learning_rate=0.001
)

# Use with PyTorch Lightning
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

# Data loaders
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
val_dataset = CIFAR10(root='./data', train=False, download=True, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Train model
trainer = pl.Trainer(max_epochs=10, accelerator="auto")
trainer.fit(model, train_loader, val_loader)
```

## Advanced Model Creation

For more complex models, you might want to implement additional features:

### 1. Adding Skip and Feedback Connections

```python
from dynvision.model_components import Skip, Feedback

def _define_architecture(self):
    # ... existing code ...
    
    # Add skip connection from layer1 to layer2
    self.addskip_layer2 = Skip(
        source=self.layer1,
        auto_adapt=True,
    )
    
    # Add feedback connection from layer2 to layer1
    self.addfeedback_layer1 = Feedback(
        source=self.layer2,
        auto_adapt=True,
    )
    
    # ... rest of the architecture ...
```

### 2. Adding Supralinearity

```python
from dynvision.model_components import SupraLinearity

def _define_architecture(self):
    # ... existing code ...
    
    # Add supralinear activation
    self.supralin = SupraLinearity(
        n=1.5,  # Exponent for supralinearity
        requires_grad=False
    )
    
    # ... rest of the architecture ...
```

### 3. Adding Retinal Preprocessing

```python
from dynvision.model_components import Retina

def _define_architecture(self):
    # ... existing code ...
    
    # Add retinal preprocessing
    self.retina = Retina(
        in_channels=self.n_channels,
        out_channels=16,
        mid_channels=32,
        kernel_size=9,
        bias=True
    )
    
    # ... rest of the architecture ...
```

## Building Complex Hierarchical Models

For more complex models with a hierarchical structure, you can follow the pattern used in `DyRCNNx4`:

```python
class HierarchicalRCNN(DyRCNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._define_architecture()
    
    def _define_architecture(self):
        """Define a hierarchical model with V1, V2, V4, and IT layers."""
        self.layer_names = ["V1", "V2", "V4", "IT"]
        self.layer_operations = [
            "layer",       # Apply (recurrent) convolutional layer
            "addext",      # Add external input
            "addskip",     # Add skip connection
            "addfeedback", # Add feedback connection
            "tstep",       # Apply dynamical systems ODE solver step
            "nonlin",      # Apply nonlinearity
            "supralin",    # Apply supralinearity
            "record",      # Record activations in responses dict
            "delay",       # Set and get delayed activations for next layer
            "pool",        # Apply pooling
            "norm",        # Apply normalization
        ]
        
        # Common parameters for layers
        layer_params = {
            "bias": self.bias,
            "recurrence_type": self.recurrence_type,
            "dt": self.dt,
            "tau": self.tau,
            "history_length": self.t_feedforward,
            "recurrence_delay": self.t_recurrence,
            "max_weight_init": self.max_weight_init,
            "device": self.device,
        }
        
        # Activation functions
        self.nonlin = nn.ReLU(inplace=False)
        if hasattr(self, "supralinearity") and float(self.supralinearity) != 1:
            self.supralin = SupraLinearity(
                n=float(self.supralinearity), requires_grad=False
            )
        
        # Define V1 layer
        self.V1 = RecurrentConnectedConv2d(
            in_channels=self.n_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            dim_y=self.dim_y,
            dim_x=self.dim_x,
            **layer_params
        )
        self.tau_V1 = torch.nn.Parameter(
            torch.tensor(self.tau, dtype=float),
            requires_grad=self.train_tau,
        )
        self.tstep_V1 = EulerStep(dt=self.dt, tau=self.tau_V1)
        self.pool_V1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Define V2 layer
        # ... similar pattern for V2, V4, IT layers ...
        
        # Add skip and feedback connections if enabled
        if self.skip:
            self.addskip_V4 = Skip(source=self.V1, auto_adapt=True)
            self.addskip_IT = Skip(source=self.V2, auto_adapt=True)
        
        if self.feedback:
            self.addfeedback_V1 = Feedback(source=self.V4, auto_adapt=True)
            self.addfeedback_V2 = Feedback(source=self.IT, auto_adapt=True)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.IT.out_channels, self.n_classes),
        )
    
    def reset(self):
        """Reset all internal states."""
        # Reset all layers and components
        for layer_name in self.layer_names:
            getattr(self, layer_name).reset()
            getattr(self, f"tstep_{layer_name}").reset()
```

## Best Practices for Custom Models

When creating custom models, follow these best practices:

### 1. Proper Initialization

Always implement proper initialization in your model:

```python
def _init_parameters(self):
    """Initialize parameters for the model."""
    super()._init_parameters()
    
    # Initialize weights with small values
    for layer_name in self.layer_names:
        layer = getattr(self, layer_name)
        if hasattr(layer, "conv.weight"):
            nn.init.trunc_normal_(layer.conv.weight, mean=0.0, std=0.004)
    
    # Initialize classifier
    nn.init.trunc_normal_(self.classifier[-1].weight, mean=0.0, std=0.004)
    nn.init.constant_(self.classifier[-1].bias, 0)
```

### 2. Reset Method

Implement a comprehensive `reset()` method that resets all stateful components:

```python
def reset(self):
    """Reset all internal states."""
    for layer_name in self.layer_names:
        if hasattr(self, layer_name):
            getattr(self, layer_name).reset()
        if hasattr(self, f"tstep_{layer_name}"):
            getattr(self, f"tstep_{layer_name}").reset()
    
    # Reset additional components
    if hasattr(self, "retina"):
        self.retina.reset()
    if hasattr(self, "input_adaption"):
        self.input_adaption.reset()
```

### 3. Setup Method

Implement a `setup()` method to handle initialization that needs to happen before training:

```python
def setup(self, stage: Optional[str]) -> None:
    """Set up the model for training or evaluation."""
    if stage == "fit" and self.feedback:
        # Initialize feedback connections with a forward pass
        x = torch.randn((1, *self.input_dims), device=self.device)
        y = self.forward(x, store_responses=False)
        self.reset()
    
    super().setup(stage)
```

### 4. Dynamic Layer Properties

Use properties to dynamically compute layer dimensions when needed:

```python
@property
def v1_output_shape(self):
    return (
        self.V1.out_channels,
        self.dim_y // self.V1.stride // self.pool_V1.stride,
        self.dim_x // self.V1.stride // self.pool_V1.stride
    )
```

## Common Customizations

### 1. Custom Recurrence Types

You can implement custom recurrence types by creating a new recurrence class:

```python
class CustomRecurrence(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size,
        bias=False,
        max_weight_init=0.05,
        **kwargs
    ):
        super().__init__()
        self.max_weight_init = max_weight_init
        
        # Implement your custom recurrence
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            bias=bias
        )
    
    def forward(self, x):
        return self.conv(x)
    
    def _init_parameters(self):
        nn.init.uniform_(
            self.conv.weight, a=-self.max_weight_init, b=self.max_weight_init
        )
```

### 2. Custom Dynamics Solvers

You can implement custom dynamics solvers by subclassing `BaseSolver`:

```python
from dynvision.model_components import BaseSolver

class CustomSolver(BaseSolver):
    def __init__(
        self,
        dt=2.0,
        tau=5.0,
        stability_check=False
    ):
        super().__init__(dt=dt, tau=tau, stability_check=stability_check)
    
    def forward(self, x, h=None):
        # Implement your custom dynamics equation
        if x is None:
            return None
        
        use_own_hidden_state = h is None
        if use_own_hidden_state and self.hidden_state is not None:
            h = self.hidden_state
        
        # Custom dynamics implementation
        if h is None:
            y = self.dt / self.tau * x
        else:
            # Your custom update rule
            y = h + self.dt / self.tau * (x - h) * torch.sigmoid(x)
        
        if self.stability_check:
            check_stability(y)
        
        if use_own_hidden_state:
            self.hidden_state = y
        
        return y
```

### 3. Custom Loss Functions

You can implement custom loss functions by subclassing `BaseLoss`:

```python
from dynvision.losses import BaseLoss
import torch
import torch.nn.functional as F

class SpatioTemporalLoss(BaseLoss):
    def __init__(self, weight=1.0, ignore_index=-100):
        super().__init__(weight=weight, ignore_index=ignore_index)
    
    def forward(self, predictions, targets):
        outputs, responses = predictions
        
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(
            outputs, targets, ignore_index=self.ignore_index
        )
        
        # Additional temporal smoothness loss
        temporal_loss = 0.0
        if responses and "V1" in responses:
            v1_response = responses["V1"]
            if len(v1_response.shape) > 3:  # Has temporal dimension
                temp_diff = torch.diff(v1_response, dim=1)
                temporal_loss = torch.mean(temp_diff**2)
        
        return ce_loss + 0.1 * temporal_loss
```

## Example: Building a DyRCNN with Different Configuration

Here's an example of a DyRCNN variant that uses local recurrence and incorporates both skip and feedback connections:

```python
import torch
import torch.nn as nn
from dynvision.model_components import RecurrentConnectedConv2d, EulerStep
from dynvision.model_components import Skip, Feedback, SupraLinearity
from dynvision.models import DyRCNN

class DyRCNNx3(DyRCNN):
    """
    A three-layer DyRCNN with local recurrence and both skip and feedback connections.
    """
    def __init__(self, **kwargs):
        # Set default values for this model
        defaults = {
            "recurrence_type": "local",
            "skip": True,
            "feedback": True,
            "supralinearity": 1.2,
        }
        # Override defaults with provided values
        for key, value in defaults.items():
            if key not in kwargs:
                kwargs[key] = value
        
        super().__init__(**kwargs)
        self._define_architecture()
    
    def _define_architecture(self):
        """Define a three-layer architecture with local recurrence."""
        self.layer_names = ["V1", "V2", "IT"]
        # Define operations order within layer
        self.layer_operations = [
            "layer",       # Apply (recurrent) convolutional layer
            "addskip",     # Add skip connection
            "addfeedback", # Add feedback connection
            "tstep",       # Apply dynamical systems ODE solver step
            "nonlin",      # Apply nonlinearity
            "supralin",    # Apply supralinearity
            "record",      # Record activations in responses dict
            "delay",       # Set and get delayed activations for next layer
            "pool",        # Apply pooling
        ]
        
        # Activation functions
        self.nonlin = nn.ReLU(inplace=False)
        self.supralin = SupraLinearity(
            n=float(self.supralinearity), requires_grad=False
        )
        
        # Common parameters for all layers
        layer_params = {
            "bias": self.bias,
            "recurrence_type": self.recurrence_type,
            "dt": self.dt,
            "tau": self.tau,
            "history_length": self.t_feedforward,
            "recurrence_delay": self.t_recurrence,
            "max_weight_init": self.max_weight_init,
            "device": self.device,
        }
        
        # Define V1 layer
        self.V1 = RecurrentConnectedConv2d(
            in_channels=self.n_channels,
            out_channels=48,
            kernel_size=5,
            stride=1,
            dim_y=self.dim_y,
            dim_x=self.dim_x,
            **layer_params
        )
        self.tau_V1 = torch.nn.Parameter(
            torch.tensor(self.tau, dtype=float),
            requires_grad=self.train_tau,
        )
        self.tstep_V1 = EulerStep(dt=self.dt, tau=self.tau_V1)
        self.pool_V1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Define V2 layer
        self.V2 = RecurrentConnectedConv2d(
            in_channels=48,
            out_channels=96,
            kernel_size=3,
            stride=1,
            dim_y=self.dim_y // 2,  # After pooling from V1
            dim_x=self.dim_x // 2,  # After pooling from V1
            **layer_params
        )
        self.tau_V2 = torch.nn.Parameter(
            torch.tensor(self.tau, dtype=float),
            requires_grad=self.train_tau,
        )
        self.tstep_V2 = EulerStep(dt=self.dt, tau=self.tau_V2)
        self.pool_V2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Define IT layer
        self.IT = RecurrentConnectedConv2d(
            in_channels=96,
            out_channels=192,
            kernel_size=3,
            stride=1,
            dim_y=self.dim_y // 4,  # After pooling from V1 and V2
            dim_x=self.dim_x // 4,  # After pooling from V1 and V2
            **layer_params
        )
        self.tau_IT = torch.nn.Parameter(
            torch.tensor(self.tau, dtype=float),
            requires_grad=self.train_tau,
        )
        self.tstep_IT = EulerStep(dt=self.dt, tau=self.tau_IT)
        
        # Skip connections
        if self.skip:
            self.addskip_V2 = Skip(source=self.V1, auto_adapt=True)
            self.addskip_IT = Skip(source=self.V2, auto_adapt=True)
        
        # Feedback connections
        if self.feedback:
            self.addfeedback_V1 = Feedback(source=self.V2, auto_adapt=True)
            self.addfeedback_V2 = Feedback(source=self.IT, auto_adapt=True)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(192, self.n_classes),
        )
    
    def reset(self):
        """Reset all internal states."""
        for layer_name in self.layer_names:
            getattr(self, layer_name).reset()
            getattr(self, f"tstep_{layer_name}").reset()
```

## Integration with Snakemake Workflows

Custom models can be easily integrated with DynVision's Snakemake workflows:

1. Place your custom model in the `dynvision/models/` directory
2. Import it in `dynvision/models/__init__.py`
3. Use it in workflows with the `model_name` parameter:

```bash
snakemake -j1 experiment --config model_name=DyRCNNx3 experiment=contrast
```

## Conclusion

Creating custom models in DynVision allows you to explore different architectures while leveraging the toolbox's biologically plausible components and training infrastructure. By following the patterns and best practices described in this guide, you can create complex models with various recurrence types, skip and feedback connections, and custom dynamics.

For more examples, see the pre-implemented models in the `dynvision/models/` directory.
