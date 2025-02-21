from types import SimpleNamespace

import numpy as np


def check_weights(model, message="", min=-2, max=2):
    layer_names = model.state_dict().keys()
    layer_names = [name.rstrip(".weight") for name in layer_names]
    weight_info = {}
    contain_nan = False

    for layer in model.state_dict().keys():
        layer_name = layer.rstrip(".weight")
        weights = model.state_dict()[layer].numpy().flatten()
        n_weights = len(weights)
        min_value = np.nanmin(weights)
        max_value = np.nanmax(weights)
        norm = np.linalg.norm(weights)
        weight_info[layer_name] = SimpleNamespace(
            min=min_value, max=max_value, norm=norm
        )

        is_nan = ~np.isfinite(weights)
        is_large = np.abs(weights) > max
        is_small = np.abs(weights) < min
        is_bad = is_nan.any() or is_small.any() or is_large.any()

        if is_bad:
            print(message)
            print(f"Layer: {layer_name}")
            if is_large.any():
                n_large = np.sum(is_large.astype(int))
                print(f"\t{n_large}/{n_weights} large weights (>{max:2f})")
            if is_small.any():
                n_small = np.sum(is_small.astype(int))
                print(f"\{n_small}/{n_weights} small weights (<{min:3f})")
            if is_nan.any():
                n_nans = np.sum(is_nan.astype(int))
                print(f"\t{n_nans}/{n_weights} NaN weights")
                contain_nan = True

    return weight_info, contain_nan
