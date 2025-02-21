from pathlib import Path

import torch
import yaml

from dynvision import losses


def get_loss_function(loss_name, loss_config={}):
    if isinstance(loss_config, Path):
        with open(loss_config, "r") as f:
            loss_config = yaml.safe_load(f)

    if hasattr(losses, loss_name):
        return getattr(losses, loss_name)(**loss_config)

    elif hasattr(torch.nn, loss_name):
        return getattr(torch.nn, loss_name)()

    else:
        raise ValueError(f"Invalid loss function: {loss_name}")


def mask_non_labels(outputs, label_indices, non_label_index=-1, selector="!="):
    # make sure outputs and label_indices are flattened
    if selector == "!=":
        mask = torch.where(label_indices != non_label_index)[0]
    elif selector == "==":
        mask = torch.where(label_indices == non_label_index)[0]
    else:
        raise ValueError(f"Invalid selector: {selector}. Must be '!=' or '=='.")

    label_indices = label_indices[mask].to(outputs.device)
    mask = mask.to(outputs.device)
    outputs = outputs[mask]

    return outputs, label_indices


def calculate_loss(
    criterion,
    outputs,
    label_indices,
    responses=None,
):

    *_, n_classes = outputs.shape

    # Flatten the outputs and label_indices over batch_size and timesteps
    outputs = outputs.reshape(-1, n_classes)
    label_indices = label_indices.reshape(-1)

    # # Exclude label_indices that are -1 from the loss calculations
    # outputs, label_indices = mask_non_labels(
    #     outputs, label_indices, non_label_index=-1, selector="!="
    # )

    # # Get the classifier responses for -2 label indices
    # void_outputs, _ = mask_non_labels(
    #     outputs, label_indices, non_label_index=-2, selector="=="
    # )

    # # Exclude label_indices that are -2 from the classification loss calculations
    # outputs, label_indices = mask_non_labels(
    #     outputs, label_indices, non_label_index=-2, selector="!="
    # )

    # # Initialize loss
    # loss = torch.tensor(0.0, requires_grad=True)

    # # Calculate the void response loss
    # if len(void_outputs):
    #     expected_init_loss = torch.ln(n_classes)
    #     void_loss_factor = 0.1

    #     void_criterion = get_loss_function("MeanSquaredActivationLoss")
    #     void_loss = void_criterion(void_outputs, target=0)
    #     loss += void_loss / expected_init_loss * void_loss_factor

    # Calculate the loss contributions
    if not isinstance(criterion, list):
        criterion = [criterion]

    loss_values = []
    for criterion_fn in criterion:
        if isinstance(criterion_fn, tuple):
            criterion_fn, weight = criterion_fn
        else:
            weight = 1

        loss_value = weight * criterion_fn((outputs, responses), label_indices)
        # print("Loss Value: ", criterion_fn, loss_value)

        loss_values += [loss_value]

    loss = sum(loss_values)

    if torch.isnan(loss):
        print("Loss Values:", criterion, loss_values)
        print("Output contains NaNs:", torch.isnan(outputs).any())
        raise ValueError("Loss value is NaN")

    return loss
