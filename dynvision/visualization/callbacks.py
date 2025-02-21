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
            label_set = df.label_set.unique()[0]
            plot_df = df[(df.label_set == label_set)]
            fig, ax = plot_classifier_responses(plot_df)

            trainer.logger.experiment.log(
                {"validation/classifier_response": wandb.Image(fig)}
            )
        return None


class MonitorTimescales(pl.Callback):
    def on_validation_end(self, trainer, model):

        print("\nCalculating Autocorrelation Timescales ...")

        tau_dict = self.calculate_layer_timescales(model.responses)

        fig = self.plot_layer_timescales(tau_dict)

        # Log the custom metric to Weights & Biases
        # trainer.logger.experiment.log({"val/custom_metric": numeric_metric})

        trainer.logger.experiment.log(
            {"validation/autocorrelation_timescales": wandb.Image(fig)}
        )
        return None

    def plot_layer_timescales(self, timescale_dict):
        fig, axes = plt.subplots(
            nrows=len(timescale_dict.keys()),
            figsize=(10, 3 * len(timescale_dict)),
            sharex=True,
        )

        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        for ax, layer_name in zip(axes, timescale_dict.keys()):
            values = timescale_dict[layer_name]
            values = values[torch.isfinite(values)]
            values = values[values >= torch.tensor([0], device=values.device)]

            values = values.cpu().numpy()

            sns.histplot(values, bins=50, edgecolor="white", ax=ax)
            ax.axvline(np.mean(values), color="0.5", linestyle=":")
            ax.set_ylabel(layer_name)
            ax.set_xlabel("Tau [time steps]")
            sns.despine(left=True, ax=ax)

        return fig

    def calculate_layer_timescales(self, response_dict, plot=False):
        tau_dict = {}

        for layer_name, layer_response in response_dict.items():
            if isinstance(layer_response, list):
                layer_response = layer_response[-1]

            layer_response = layer_response.detach()
            layer_response = layer_response.permute(
                1, 0, *range(2, layer_response.dim())
            )

            # average over the spatial dimensions
            layer_response = layer_response.mean(dim=(-2, -1))

            n_timesteps, *_ = layer_response.shape
            unit_responses = layer_response.reshape(n_timesteps, -1)

            # Remove any dimensions that contain only zeros
            non_zero_mask = torch.any(unit_responses != 0, dim=0)
            unit_responses = unit_responses[:, non_zero_mask]
            non_zero_mask = torch.any(unit_responses != 0, dim=1)
            unit_responses = unit_responses[non_zero_mask, :]

            tau_dict[layer_name] = torch.zeros(
                unit_responses.shape[1], dtype=torch.int, device=unit_responses.device
            )

            for i, unit_response in tqdm(
                enumerate(unit_responses.T),
                total=unit_responses.shape[1],
                desc=layer_name,
            ):
                tau = self.calculate_autocorrelation_timescale(
                    unit_response, plot=plot
                )
                try:
                    tau_dict[layer_name][i] = tau
                except Exception as e:
                    print(e)
                    print("Tau: ", tau, type(tau))
                    tau_dict[layer_name][i] = torch.tensor([-1])

        return tau_dict

    def calculate_autocorrelation_timescale(self, timeseries, plot=True):
        """
        Calculate the autocorrelation timescale for a timeseries of response activity.

        Parameters:
            timeseries (array-like): The response activity timeseries of a neural unit.

        Returns:
            tau (float): The autocorrelation timescale.
        """

        if (timeseries == 0).all():
            return torch.tensor([-1])

        # Step 1: removing NaNs
        if torch.isnan(timeseries).any():
            print(
                "Removing",
                torch.sum(torch.isnan(timeseries)),
                "/",
                len(timeseries),
                "NaNs",
            )
            timeseries = timeseries(torch.isfinite(timeseries))

        # Step 2: Z-score the firing rate
        mean = torch.mean(timeseries)
        std = torch.std(timeseries)
        z_scored_rate = (timeseries - mean) / std

        # Step 3: Calculate the autocorrelation of the z-scored signal
        autocorr = torch.nn.functional.conv1d(
            z_scored_rate.view(1, 1, -1),
            z_scored_rate.view(1, 1, -1),
            padding=z_scored_rate.size(0) - 1,
        )
        autocorr = autocorr.view(-1)[autocorr.size(2) // 2 :]
        autocorr /= torch.max(autocorr)

        # Step 3.5: Handle undefined correlation values
        autocorr[torch.isnan(autocorr) | torch.isinf(autocorr)] = 0

        # Step alt 4: Find the first lag where the autocorrelation drops below 1/e
        threshold = 1 / torch.exp(
            torch.tensor([1.0], device=autocorr.device, dtype=autocorr.dtype)
        )
        condition = (autocorr <= threshold).to(torch.long)
        best_tau = torch.argmax(condition)

        # # Step 4: Define the exponential decay function
        # def exponential_decay(x, a, tau, b):
        #     return a * torch.exp(-x / tau) + b

        # # Fit the exponential function to the autocorrelation function
        # x_vals = torch.arange(len(autocorr), device=autocorr.device, dtype=autocorr.dtype)
        # tau_values = torch.linspace(1, 31.0, 61, device=autocorr.device)  # Search space for tau
        # best_loss = float('inf')
        # best_params = None

        # for tau in tau_values:
        #     # Compute the transformed x_data for the current tau
        #     exp_term = torch.exp(-x_vals / tau)

        #     # Set up the matrix A for least squares: A @ [a, b] = y
        #     # A has columns [exp_term, ones]
        #     A = torch.stack([exp_term, torch.ones_like(exp_term)], dim=1)

        #     # Solve for [a, b] using least squares: [a, b] = (A^T A)^(-1) A^T y
        #     params = torch.linalg.lstsq(A, autocorr).solution
        #     a, b = params

        #     # Compute the loss for this tau value
        #     y_fit = exponential_decay(x_vals, a, tau, b)
        #     loss = torch.sum((y - y_fit) ** 2)

        #     # Check if this is the best tau found so far
        #     if loss < best_loss:
        #         best_loss = loss
        #         best_tau = tau
        #         y_vals = y_fit

        if plot:
            # Plot the autocorrelation function
            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, autocorr, label="Autocorrelation", color="k")
            plt.plot(x_vals, y_vals, label="Fit", color="g", linestyle="--")
            plt.text(
                len(autocorr) // 2,
                max(autocorr) / 2,
                f"Tau: {tau:.2f}",
                fontsize=16,
                color="k",
                ha="center",
            )
            plt.xlabel("Lag")
            plt.ylabel("Autocorrelation")
            plt.title("Autocorrelation Function")
            plt.legend()
            plt.show()

        return best_tau
