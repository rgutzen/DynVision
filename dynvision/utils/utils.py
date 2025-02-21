import inspect
import random
import re
import warnings
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from tqdm import tqdm as progress_bar
import yaml

from dynvision.project_paths import project_paths

types_int = [
    "id",
    "index",
    "sample_index",
    "times_index",
    "class_index",
    "label_index",
    "guess_index",
]
types_float = ["response"]
types_bool = []
types_str = ["label_set"]

dtypes = (
    dict.fromkeys(types_int, int)
    | dict.fromkeys(types_float, float)
    | dict.fromkeys(types_bool, bool)
    | dict.fromkeys(types_str, str)
)


def load_df(path, dtypes=dtypes):
    df = pd.read_csv(path, dtype=dtypes)
    df.drop(
        df.columns[df.columns.str.contains("unnamed", case=False)],
        axis=1,
        inplace=True,
    )
    return df


_true_set = {"yes", "true", "t", "y", "1"}
_false_set = {"no", "false", "f", "n", "0"}


def str_to_bool(value, raise_exc=False):
    if isinstance(value, str):
        value = value.lower()
        if value in _true_set:
            return True
        if value in _false_set:
            return False
    elif isinstance(value, bool):
        return value
    if raise_exc:
        raise ValueError('Expected "%s"' % '", "'.join(_true_set | _false_set))
    return None


def set_seed(seed):
    if isinstance(seed, str):
        seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed, workers=True)
    return None


def alias_kwargs(**aliases):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for alias, original in aliases.items():
                if alias in kwargs and original not in kwargs:
                    kwargs[original] = kwargs.pop(alias)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def parse_kwargs(kwargs: list):
    params = {}
    key = None
    for element in kwargs:
        if element[:2] == "--":
            key = element.lstrip("-")
            params[key] = []
        else:
            params[key] += [element]
    for k, v in params.items():
        if len(v) == 1:
            params[k] = v[0]
    return params


def filter_kwargs(function, kwargs):
    # Filter the argument names that the function accepts
    args = inspect.signature(function).parameters.keys()
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in args}
    return filtered_kwargs


def identity(*args, **kwargs):
    return args[0]


def tqdm(*args, **kwargs):
    if project_paths.iam_on_cluster():
        # return identity(*args, **kwargs)
        return progress_bar(*args, **kwargs)
    else:
        return progress_bar(*args, **kwargs)


def path_to_index(path):
    if not isinstance(path, str) or isinstance(path, Path):
        warnings.warn(f"Invalid path type: {path}")
        return -1
    name = Path(path).stem
    index = int(name.split("_")[-1])
    return index


@dataclass
class FlatIndices:
    chan_flat: np.ndarray
    x_flat: np.ndarray
    y_flat: np.ndarray


def get_flat_indices(dims):
    """
    dims should be a CHW tuple
    """
    num_channels, num_x, num_y = dims

    # chan flat goes like [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, ...]
    chan_flat = np.repeat(np.arange(num_channels), num_x * num_y)

    # x flat goes like [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, ...]
    x_flat = np.repeat(np.tile(np.arange(num_x), num_channels), num_y)

    # y flat goes like [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, ...]
    y_flat = np.tile(np.arange(num_y), num_x * num_channels)

    return FlatIndices(chan_flat, x_flat, y_flat)


def is_square_number(n):
    if n < 0:
        return False
    sqrt_n = int(np.sqrt(n))
    return sqrt_n * sqrt_n == n


def next_square_number(c):
    i = int(c**0.5) + 1
    return i**2


def extend_to_square_channel_number(x):
    n_channels, dim_y, dim_x = x.shape

    if is_square_number(n_channels):
        return x

    add_n = next_square_number(n_channels) - n_channels
    repeated_slices = x[-add_n::-1]  # reflecting last slices

    return torch.cat((x, repeated_slices), dim=0)


def extract_param_from_string(s, key="contrast", value_type=None):
    if value_type == int:
        match = re.search(rf"{key}=(\d+)", s)
    elif value_type == float:
        match = re.search(rf"{key}=(\d+(\.\d+)?)", s)
    elif value_type == str:
        match = re.search(rf"{key}=([a-z]+)", s)
    elif value_type is None:
        match = re.search(rf"{key}=([\da-z\.]+)", s)
        value_type = guess_type
    else:
        raise ValueError(f"Invalid value type: {value_type}")
    if match:
        return value_type(match.group(1))
    else:
        raise ValueError(f"No {key} value found in the string!")


def replace_param_in_string(s, key="contrast", value_type=None, new_value="*"):
    if value_type == int:
        match = re.search(rf"{key}=(\d+)", s)
    elif value_type == float:
        match = re.search(rf"{key}=(\d+(\.\d+)?)", s)
    elif value_type == str:
        match = re.search(rf"{key}=([a-z]+)", s)
    elif value_type is None:
        match = re.search(rf"{key}=([\da-z\.]+)", s)
    else:
        raise ValueError(f"Invalid value type: {value_type}")
    if match:
        return s.replace(match.group(0), f"{key}={new_value}")
    else:
        raise ValueError(f"No {key} value found in the string!")


def load_config(path: Path) -> dict:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def guess_type(string):
    try:
        out = int(string)
    except:
        try:
            out = float(string)
        except:
            out = str(string)
            if out == "None":
                out = None
            elif out == "True":
                out = True
            elif out == "False":
                out = False
    return out


def str2dict(string, assigner=":", separator=","):
    """
    Transforms a str(dict) back to dict
    """
    if string[0] == "{":
        string = string[1:]
    if string[-1] == "}":
        string = string[:-1]
    my_dict = {}
    # list or tuple values
    brackets = [delimiter for delimiter in ["[", "]", "(", ")"] if delimiter in string]
    if len(brackets):
        for kv in string.split(f"{brackets[1]}{separator}"):
            k, v = kv.split(assigner)
            v = v.replace(brackets[0], "").replace(brackets[1], "")
            values = [guess_type(val) for val in v.split(separator)]
            if len(values) == 1:
                values = values[0]
            my_dict[k.strip()] = values
    # scalar values
    else:
        for kv in string.split(separator):
            k, v = kv.split(assigner)
            my_dict[k.strip()] = guess_type(v.strip())
    return my_dict


def parse_string2dict(kwargs_str, **kwargs):
    if type(kwargs_str) == list:
        if len(kwargs_str) == 0:
            return {}
        elif len(kwargs_str) == 1:
            kwargs = kwargs_str[0]
        else:
            kwargs = "".join(kwargs_str)[1:-1]
    else:
        kwargs = str(kwargs_str)

    if guess_type(kwargs) is None:
        return {}

    kwargs = kwargs.strip("{}")
    kwargs = kwargs.replace("'", "")

    my_dict = {}

    # match all nested dicts
    pattern = re.compile("[\w\s]+:{[^}]*},*")
    for match in pattern.findall(kwargs):
        nested_dict_name, nested_dict = match.split(":{")
        nested_dict = nested_dict[:-1]
        my_dict[nested_dict_name] = str2dict(nested_dict)
        kwargs = kwargs.replace(match, "")

    # match entries with word value, list value, or tuple value
    pattern = re.compile("[\w\s]+:(?:[\w\.\s\/\-\&\+]+|\[[^\]]+\]|\([^\)]+\))")
    for match in pattern.findall(kwargs):
        my_dict.update(str2dict(match))

    return my_dict
