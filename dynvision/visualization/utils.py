import logging
import sys
from pathlib import Path
import pandas as pd
import torch
from copy import deepcopy

import matplotlib.pyplot as plt

from dynvision.utils.utils import (
    extract_param_from_string,
    load_df,
    guess_type,
)


def layer_power(response):
    # returns the mean power in shape [n_features, n_timesteps]
    return response.mean(dim=list(range(2, response.dim())))


def peak_time(response):
    mean_power = layer_power(response)
    return mean_power.argmax(dim=1)


def peak_height(response):
    mean_power = layer_power(response)
    max_values, max_indices = mean_power.max(dim=1)
    return max_values


def peak_ratio(response, min_delay = 3):
    mean_power = layer_power(response)
    peak1_index = mean_power.argmax(dim=1)
    peak1_value = torch.tensor([deepcopy(mean_power[channel, i].item()) for channel, i in enumerate(peak1_index)])

    for channel, i in enumerate(peak1_index):
        mean_power[channel, i-min_delay : i+min_delay] = float('-inf')
        
    peak2_index = mean_power.argmax(dim=1)
    peak2_value = [mean_power[channel, i] for channel, i in enumerate(peak2_index)]
        
    ratio = torch.Tensor([p1/p2 if i1<i2 else p2/p1 for i1, p1, i2, p2 in zip(peak1_index, peak1_value, peak2_index, peak2_value)])
    return ratio

def calculate_accuracy(df):
    dfi = df[df.label_index != -1]
    n_correct = (dfi.guess_index == dfi.label_index).sum()
    accuracy = n_correct / len(dfi)
    return accuracy


def load_responses(
    pt_files,
    csv_files,
    data_arg_key="contrast",
    measures=["power", "peak_time", "peak_height"],
    category="rctype",
):
    dfs = []

    for pt_file, csv_file in zip(pt_files, csv_files):
        arg_value = extract_param_from_string(
            pt_file.stem, key=data_arg_key, value_type=float
        )

        cat_value = extract_param_from_string(
            pt_file.stem, key=category, value_type=None
        )

        if not arg_value == extract_param_from_string(
            csv_file.stem, key=data_arg_key, value_type=float
        ):
            raise ValueError(f"{data_arg_key} values do not match!")
        if not cat_value == extract_param_from_string(
            csv_file.stem, key=category, value_type=None
        ):
            raise ValueError(f"{category} do not match!")

        df = load_df(csv_file)
        df[data_arg_key] = arg_value
        df[category] = cat_value

        n_classes = len(df.class_index.unique())

        responses = torch.load(pt_file, map_location=torch.device("cpu"))
        # responses = {k: v.cpu() for k, v in responses.items()}
        layer_names = list(responses.keys())
        n_samples, n_timesteps, *_ = responses[layer_names[0]].shape

        for layer in layer_names:
            if "power" in measures:
                df[f"{layer}_power"] = (
                    layer_power(responses[layer])
                    .flatten()
                    .repeat_interleave(n_classes)
                )
            if "peak_time" in measures:
                df[f"{layer}_peak_time"] = peak_time(
                    responses[layer]
                ).repeat_interleave(n_classes * n_timesteps)
            if "peak_height" in measures:
                df[f"{layer}_peak_height"] = peak_height(
                    responses[layer]
                ).repeat_interleave(n_classes * n_timesteps)
            if "peak_ratio" in measures:
                df[f"{layer}_peak_ratio"] = peak_ratio(
                    responses[layer]
                ).repeat_interleave(n_classes * n_timesteps)

        dfs.append(df)

    df = pd.concat(dfs, axis=0)

    return df, layer_names


def save_plot(file_path: Path, dpi: int = 300, **kwargs):
    file_path = Path(file_path)
    file_path.parent.mkdir(exist_ok=True, parents=True)

    try:
        plt.savefig(fname=file_path, dpi=dpi, bbox_inches="tight", **kwargs)

    except Exception as e:
        logging.error(f"Failed to save plot: {e}")
        try:  # save empty plot
            plt.subplots()
            plt.savefig(fname=file_path)
        except Exception as e:
            logging.error(f"Failed to save empty plot: {e}")

    finally:
        if "ipykernel" in sys.modules:
            plt.show()
        else:
            plt.close()

    logging.info(f"Plot saved successfully at {file_path}")

    return None
