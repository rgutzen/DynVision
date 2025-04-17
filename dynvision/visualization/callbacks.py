from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import wandb
from tqdm import tqdm

from dynvision.visualization.plot_classifier_responses import (
    plot_classifier_responses,
)
from dynvision.visualization.plot_weight_distributions import (
    plot_weight_distributions,
)


class MonitorWeightDistributions(pl.Callback):
    def on_validation_end(self, trainer, model):
        state_dict = model.state_dict()

        fig, ax = plot_weight_distributions(state_dict)

        trainer.logger.experiment.log(
            {"validation/weight_distributions": wandb.Image(fig)}
        )
        return None


class MonitorClassifierResponses(pl.Callback):
    def on_validation_end(self, trainer, model):
        df = model.get_classifier_dataframe()

        if len(df):
            fig, ax = plot_classifier_responses(df)

            trainer.logger.experiment.log(
                {"validation/classifier_response": wandb.Image(fig)}
            )
        return None
