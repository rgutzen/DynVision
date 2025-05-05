# Model Evaluation

This guide explains how to evaluate models in DynVision, including both standard performance metrics and biological plausibility assessments.

## Overview

DynVision's evaluation framework goes beyond standard deep learning metrics to assess both:

1. **Task Performance**: Classification accuracy and loss
2. **Biological Plausibility**: Temporal dynamics and response properties

The evaluation process typically involves:
- Testing models on standard datasets
- Running specialized experiments to probe neural response properties
- Analyzing and visualizing the results
- Comparing models with different architectures or parameters

## Standard Evaluation

### Classification Performance

To evaluate a model's classification performance:

```bash
# Test a trained model on a dataset
snakemake -j1 test_model --config \
  model_name=DyRCNNx4 \
  model_args="{rctype:full}" \
  data_name=cifar100 \
  data_group=invertebrates \
  status=trained \
  data_loader=StandardDataLoader
```

This command runs the model on the specified test dataset and saves the results.

### Evaluation Metrics

DynVision calculates several standard metrics:

1. **Accuracy**: Percentage of correctly classified samples
2. **Loss**: Cross-entropy loss on the test set
3. **Confusion Matrix**: Distribution of predictions across classes
4. **Per-class Accuracy**: Accuracy for each individual class

### Visualizing Results

To visualize the evaluation results:

```bash
# Generate confusion matrix
snakemake -j1 plot_confusion_matrix --config \
  model_name=DyRCNNx4 \
  model_args="{rctype:full}" \
  data_name=cifar100 \
  data_group=invertebrates \
  status=trained
```

## Temporal Dynamics Evaluation

DynVision provides specialized experiments to evaluate temporal dynamics:

### 1. Basic Response Properties

Test basic temporal response properties with the `response` experiment:

```bash
# Run response experiment
snakemake -j1 experiment --config \
  experiment=response \
  model_name=DyRCNNx4 \
  model_args="{rctype:full}"
```

This experiment measures:
- Response latency (time to first response)
- Response duration
- Response magnitude
- Response patterns across layers

### 2. Contrast Response Function

Evaluate how the model responds to stimuli of varying contrast:

```bash
# Run contrast experiment
snakemake -j1 experiment --config \
  experiment=contrast \
  model_name=DyRCNNx4 \
  model_args="{rctype:full}"
```

Key metrics from this experiment:
- Contrast sensitivity
- Contrast-dependent response timing
- Layer-specific contrast thresholds

### 3. Stimulus Duration Effects

Assess how the model responds to stimuli of varying duration:

```bash
# Run duration experiment
snakemake -j1 experiment --config \
  experiment=duration \
  model_name=DyRCNNx4 \
  model_args="{rctype:full}"
```

Key metrics:
- Temporal summation properties
- Response saturation with longer stimuli
- Persistence after stimulus offset

### 4. Repetition Effects

Measure adaptation and recovery to repeated stimuli:

```bash
# Run interval experiment
snakemake -j1 experiment --config \
  experiment=interval \
  model_name=DyRCNNx4 \
  model_args="{rctype:full}"
```

Key metrics:
- Repetition suppression (reduced response to second stimulus)
- Recovery time course (how quickly response recovers with longer intervals)
- Layer-specific adaptation properties

## Comparing Models

DynVision makes it easy to compare different models or model configurations:

### Comparing Recurrence Types

To compare models with different recurrence types:

```bash
# Compare recurrence types on contrast experiment
snakemake -j1 all_experiments --config \
  experiment=contrast \
  model_args="{rctype:[full,self,depthpointwise,pointdepthwise]}"
```

### Comparing Architectures

To compare different model architectures:

```bash
# Compare architectures on contrast experiment
snakemake -j1 test_standard_models --config \
  experiment=contrast \
  model_name="[AlexNet,CorNetRT,ResNet18,DyRCNNx4]"
```

## Quantitative Analysis

DynVision stores evaluation results as CSV files and model responses as PyTorch tensors, allowing for detailed quantitative analysis:

### Analyzing Response Data

```python
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load model responses
responses = torch.load('models/DyRCNNx4/DyRCNNx4:rctype=full_0001_cifar100_trained_StimulusDuration:tsteps=100+stim=15_invertebrates_test_responses.pt')

# Extract layer responses
v1_response = responses['V1']  # Shape: [n_samples, n_timesteps, n_channels, height, width]

# Average over samples and spatial dimensions
v1_avg = v1_response.mean(dim=(0, 2, 3, 4))  # Average over all dimensions except time

# Calculate key metrics
response_magnitude = v1_avg.max().item()
baseline = v1_avg[:5].mean().item()  # Assuming first 5 timesteps are baseline
peak_time = v1_avg.argmax().item()
onset_time = ((v1_avg > (baseline + 0.1 * (response_magnitude - baseline))).nonzero()[0]).item()
offset_time = len(v1_avg) - 1 - ((v1_avg.flip(0) > (baseline + 0.1 * (response_magnitude - baseline))).nonzero()[0]).item()
response_duration = offset_time - onset_time

print(f"Response magnitude: {response_magnitude:.4f}")
print(f"Baseline: {baseline:.4f}")
print(f"Peak time: {peak_time}")
print(f"Onset time: {onset_time}")
print(f"Offset time: {offset_time}")
print(f"Response duration: {response_duration}")
```

### Comparing Biological Properties

To systematically compare models with biological data:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

# Load model responses
responses_full = torch.load('models/DyRCNNx4/DyRCNNx4:rctype=full_0001_cifar100_trained_StimulusContrast:tsteps=100+stim=15_invertebrates_test_responses.pt')
responses_self = torch.load('models/DyRCNNx4/DyRCNNx4:rctype=self_0001_cifar100_trained_StimulusContrast:tsteps=100+stim=15_invertebrates_test_responses.pt')

# Create function to extract biological metrics
def extract_bio_metrics(responses):
    metrics = {}
    
    # Process each layer
    for layer in ['V1', 'V2', 'V4', 'IT']:
        layer_response = responses[layer].mean(dim=(0, 2, 3, 4))  # Average over all dimensions except time
        
        # Calculate metrics
        metrics[f'{layer}_latency'] = ((layer_response > layer_response.mean() + layer_response.std()).nonzero()[0]).item()
        metrics[f'{layer}_magnitude'] = layer_response.max().item()
        metrics[f'{layer}_peak_time'] = layer_response.argmax().item()
    
    # Calculate layer timing relationships
    metrics['v1_to_v2_delay'] = metrics['V2_latency'] - metrics['V1_latency']
    metrics['v2_to_v4_delay'] = metrics['V4_latency'] - metrics['V2_latency']
    metrics['v4_to_it_delay'] = metrics['IT_latency'] - metrics['V4_latency']
    
    return metrics

# Extract metrics for each model
metrics_full = extract_bio_metrics(responses_full)
metrics_self = extract_bio_metrics(responses_self)

# Biological reference data (from Groen et al., 2022)
bio_reference = {
    'v1_to_v2_delay': 10.5,  # ms
    'v2_to_v4_delay': 15.3,  # ms
    'v4_to_it_delay': 12.1,  # ms
}

# Create comparison table
comparison = pd.DataFrame({
    'Measure': list(bio_reference.keys()),
    'Biological': list(bio_reference.values()),
    'Full Recurrence': [metrics_full[k] * 2 for k in bio_reference.keys()],  # Convert model timesteps to ms
    'Self Recurrence': [metrics_self[k] * 2 for k in bio_reference.keys()]
})

print(comparison)

# Calculate biological similarity score (lower is better)
full_diff = np.mean(np.abs(np.array(list(bio_reference.values())) - np.array([metrics_full[k] * 2 for k in bio_reference.keys()])))
self_diff = np.mean(np.abs(np.array(list(bio_reference.values())) - np.array([metrics_self[k] * 2 for k in bio_reference.keys()])))

print(f"Biological similarity score (Full): {full_diff:.2f}")
print(f"Biological similarity score (Self): {self_diff:.2f}")
```

## Biological Alignment Assessment

To systematically evaluate how well a model aligns with biological data, DynVision can calculate several key metrics:

### Response Latency Progression

Biological visual systems show characteristic latency progressions across areas:

```python
# Dictionary of biological response latencies (ms) based on Groen et al. (2022)
bio_latencies = {
    'V1': 40,
    'V2': 50,
    'V4': 65,
    'IT': 80
}

# Extract model latencies (in timesteps)
model_latencies = {}
for layer in ['V1', 'V2', 'V4', 'IT']:
    layer_response = responses[layer].mean(dim=(0, 2, 3, 4))
    threshold = layer_response.mean() + layer_response.std()
    latency = ((layer_response > threshold).nonzero()[0]).item()
    model_latencies[layer] = latency * 2  # Convert to ms (assuming dt=2ms)

# Calculate correlation between model and biological latencies
latency_correlation = np.corrcoef(
    [bio_latencies[k] for k in ['V1', 'V2', 'V4', 'IT']],
    [model_latencies[k] for k in ['V1', 'V2', 'V4', 'IT']]
)[0, 1]

print(f"Latency progression correlation: {latency_correlation:.4f}")
```

### Contrast Response Function Shape

Biological neurons show characteristic contrast response functions:

```python
# Extract contrast response for V1
contrasts = [0.2, 0.4, 0.6, 0.8, 1.0]
v1_contrast_responses = []

for contrast in contrasts:
    response = torch.load(f'models/DyRCNNx4/DyRCNNx4:rctype=full_0001_cifar100_trained_StimulusContrast:tsteps=100+stim=15+contrast={contrast}_invertebrates_test_responses.pt')
    v1_contrast_response = response['V1'].mean(dim=(0, 2, 3, 4))
    v1_contrast_responses.append(v1_contrast_response.max().item())

# Fit Naka-Rushton function (common model for contrast response)
from scipy.optimize import curve_fit

def naka_rushton(c, rmax, c50, n):
    return rmax * (c**n) / (c**n + c50**n)

params, _ = curve_fit(naka_rushton, contrasts, v1_contrast_responses, bounds=([0, 0, 0], [np.inf, 1, 10]))
rmax, c50, n = params

# Typical biological parameters
bio_c50 = 0.3  # Half-saturation contrast
bio_n = 2.0    # Exponent

print(f"Model contrast response parameters: Rmax={rmax:.4f}, C50={c50:.4f}, n={n:.4f}")
print(f"Biological reference: C50={bio_c50}, n={bio_n}")
```

### Response Pattern Analysis

To analyze temporal response patterns:

```python
# Extract response patterns
response = torch.load('models/DyRCNNx4/DyRCNNx4:rctype=full_0001_cifar100_trained_StimulusDuration:tsteps=100+stim=25_invertebrates_test_responses.pt')

# Check for key biological response patterns
def check_response_patterns(response):
    patterns = {}
    
    # Extract layer responses
    for layer in ['V1', 'V2', 'V4', 'IT']:
        layer_response = response[layer].mean(dim=(0, 2, 3, 4))
        
        # Sustained vs. transient response
        peak_value = layer_response.max().item()
        peak_index = layer_response.argmax().item()
        sustained_value = layer_response[25:35].mean().item()  # Later portion of response
        transience_index = (peak_value - sustained_value) / peak_value
        
        patterns[f'{layer}_transience'] = transience_index
        
        # Presence of offset response
        stimulus_end = 25  # Assuming stimulus ends at timestep 25
        post_stimulus = layer_response[stimulus_end:stimulus_end+10]
        baseline = layer_response[:5].mean().item()
        has_offset_response = (post_stimulus.max().item() - baseline) > 0.1 * (peak_value - baseline)
        
        patterns[f'{layer}_offset_response'] = has_offset_response
    
    return patterns

response_patterns = check_response_patterns(response)
print(response_patterns)
```

## Model Evaluation Workflow

DynVision's Snakemake workflow provides a complete model evaluation pipeline:

### Basic Workflow

```
# Initialize model
↓
# Train model
↓
# Test on standard dataset
↓
# Run specialized experiments
↓
# Analyze results
↓
# Compare with other models/configurations
```

### All-in-One Command

To run the complete evaluation pipeline:

```bash
# Complete evaluation pipeline for a model
snakemake -j1 all_experiments --config \
  model_name=DyRCNNx4 \
  model_args="{rctype:full}" \
  seed=0001 \
  data_name=cifar100 \
  data_group=invertebrates \
  experiment="[response,contrast,duration,interval]"
```

## Evaluating Custom Models

To evaluate a custom model, you need to:

1. Implement the model following DynVision's model architecture
2. Add it to `dynvision/models/__init__.py`
3. Run the evaluation commands with your model name

```bash
# Evaluate a custom model
snakemake -j1 experiment --config \
  experiment=contrast \
  model_name=MyCustomModel
```

## Best Practices for Evaluation

### 1. Set Fixed Random Seeds

For reproducible evaluations, always set fixed random seeds:

```bash
# Set random seed for reproducibility
snakemake -j1 experiment --config seed=0001
```

### 2. Use Multiple Test Runs

For robust evaluations, average results over multiple runs:

```bash
# Run multiple evaluations with different seeds
for seed in 0001 0002 0003 0004 0005; do
  snakemake -j1 experiment --config \
    experiment=contrast \
    model_name=DyRCNNx4 \
    seed=$seed
done
```

### 3. Perform Control Experiments

Include control experiments for comparison:

```bash
# Run feedforward-only experiment as control
snakemake -j1 experiment --config \
  experiment=responseffonly \
  model_name=DyRCNNx4 \
  model_args="{rctype:full}"
```

### 4. Compare with Biological Data

When possible, compare results with biological data:

```bash
# Plot model results alongside biological data
python scripts/plot_bio_comparison.py \
  --model_responses models/DyRCNNx4/responses.pt \
  --bio_data references/groen_2022_data.csv
```

## Advanced Evaluation Techniques

### 1. Adversarial Robustness

Test model robustness to adversarial examples:

```python
import torch
import torch.nn.functional as F

def generate_adversarial_example(model, x, y, epsilon=0.01, iterations=10):
    """Generate adversarial example using PGD attack."""
    x_adv = x.clone().detach().requires_grad_(True)
    
    for _ in range(iterations):
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs[:, -1], y)
        loss.backward()
        
        grad_sign = x_adv.grad.sign()
        x_adv = x_adv + epsilon * grad_sign
        x_adv = torch.clamp(x_adv, 0, 1)
        x_adv = x_adv.detach().requires_grad_(True)
    
    return x_adv

# Test model on adversarial examples
def test_adversarial_robustness(model, dataloader, epsilon=0.01):
    model.eval()
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        adv_inputs = generate_adversarial_example(model, inputs, targets, epsilon)
        outputs = model(adv_inputs)
        _, predicted = outputs[:, -1].max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return correct / total
```

### 2. Occlusion Testing

Test model robustness to occlusions:

```python
def test_occlusion_robustness(model, dataloader, occlusion_size=10):
    """Test model performance with occluded images."""
    model.eval()
    results = []
    
    for inputs, targets in dataloader:
        batch_size, t, c, h, w = inputs.shape
        
        # Create occlusion masks at different positions
        for i in range(0, h - occlusion_size, occlusion_size):
            for j in range(0, w - occlusion_size, occlusion_size):
                occluded_inputs = inputs.clone()
                occluded_inputs[:, :, :, i:i+occlusion_size, j:j+occlusion_size] = 0
                
                outputs = model(occluded_inputs)
                _, predicted = outputs[:, -1].max(1)
                accuracy = predicted.eq(targets).float().mean().item()
                
                results.append({
                    'position_x': j,
                    'position_y': i,
                    'accuracy': accuracy
                })
    
    return pd.DataFrame(results)
```

### 3. Representational Similarity Analysis

Compare model representations with biological data:

```python
from scipy.stats import spearmanr
import numpy as np

def compute_rdm(activations):
    """Compute Representational Dissimilarity Matrix."""
    n_samples = activations.shape[0]
    rdm = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            # Correlation distance
            corr = np.corrcoef(activations[i].flatten(), activations[j].flatten())[0, 1]
            rdm[i, j] = rdm[j, i] = 1 - corr
    
    return rdm

def compare_rdms(model_rdm, bio_rdm):
    """Compare model and biological RDMs using Spearman correlation."""
    # Flatten the upper triangular part of the RDMs
    model_rdm_flat = model_rdm[np.triu_indices_from(model_rdm, k=1)]
    bio_rdm_flat = bio_rdm[np.triu_indices_from(bio_rdm, k=1)]
    
    # Compute Spearman correlation
    correlation, p_value = spearmanr(model_rdm_flat, bio_rdm_flat)
    
    return correlation, p_value
```

## Conclusion

Effective model evaluation in DynVision goes beyond standard accuracy metrics to include assessment of biological plausibility. By combining traditional deep learning evaluation with specialized experiments for temporal dynamics, you can gain deeper insights into how well your models capture properties of biological vision systems.

For more information on specific evaluation methods and metrics, see the [API Reference](../reference/evaluation.md).
