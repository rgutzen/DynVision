import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from dynvision.utils.utils import load_df, parse_kwargs

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs="+", type=Path)
CLI.add_argument("--output", nargs="?", type=Path)


if __name__ == "__main__":
    args, unknown = CLI.parse_known_args()
    # add optional addition parameters
    params = parse_kwargs(unknown)

    for i, datafile in enumerate(args.data):
        df = load_df(datafile)

        # add command line arguments with one value per dataframe
        for k, v in params.items():
            if hasattr(v, "__len__") and len(v) == len(args.data):
                df[k] = v[i]

        if i:
            full_df = pd.concat((full_df, df), axis=0, ignore_index=True)
        else:
            full_df = deepcopy(df)
        del df

    # add command line arguments with one value for all dataframes
    for k, v in params.items():
        if hasattr(v, "__len__") and len(v) == 1:
            full_df[k] = v[0]

    full_df.to_csv(args.output)
