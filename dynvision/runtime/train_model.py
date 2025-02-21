import argparse
import multiprocessing
from pathlib import Path
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import wandb

from dynvision import models
from dynvision.data.dataloader import (
    _adjust_data_dimensions,
    _adjust_label_dimensions,
    get_ffcv_dataloader,
    get_train_val_loaders,
)
from dynvision.project_paths import project_paths
from dynvision.utils.utils import (
    parse_kwargs,
    parse_string2dict,
    str_to_bool,
)
from dynvision.visualization import callbacks as custom_callbacks


parser = argparse.ArgumentParser(description="Train a model on a dataset.")
parser.add_argument("--input_model_state", type=Path, help="Path to model state")
parser.add_argument("--model_name", type=str, help="Name of model class")
parser.add_argument("--dataset_train", type=Path, help="Path to the dataset")
parser.add_argument("--dataset_val", type=Path, help="Path to the dataset")
parser.add_argument("--data_transform", type=str, default=None, help="transform func")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--output_model_state", type=Path, help="Path to save model")
parser.add_argument("--resolution", type=int, default=224, help="Image resolution")
parser.add_argument(
    "--n_timesteps", "--tsteps", type=int, default=1, help="repeat image x times"
)
parser.add_argument("--seed", type=str, help="Random seed")
parser.add_argument(
    "--check_val_every_n_epoch", type=int, default=1, help="Validation check intedvml"
)
parser.add_argument(
    "--accumulate_grad_batches", type=int, default=1, help="Gradient accumulation"
)
parser.add_argument(
    "--precision", type=str, default="bf16-mixed", help="Weight precision"
)
parser.add_argument("--profiler", type=str, default=None, help="Weight precision")
parser.add_argument(
    "--benchmark", type=str_to_bool, default=True, help="Input shape is constant?"
)
parser.add_argument(
    "--store_responses", type=int, default=0, help="Store how many responses?"
)
parser.add_argument(
    "--enable_progress_bar", type=str_to_bool, default=True, help="Show progress bar?"
)
parser.add_argument(
    "--loss",
    nargs="+",
    type=str,
    default=["CrossEntropyLoss"],
    help="Loss function name",
)
parser.add_argument("--loss_config", nargs="+", type=str, default=None)


class EarlyStoppingWithMin(pl.callbacks.EarlyStopping):
    def __init__(
        self, monitor="val_accuracy", patience=5, min_val_accuracy=0.75, **kwargs
    ):
        super().__init__(monitor=monitor, patience=patience, **kwargs)
        self.min_val_accuracy = min_val_accuracy

    def on_validation_epoch_end(self, trainer, pl_module):
        # Check if the current metric value is above the minimum threshold
        current_value = trainer.callback_metrics.get(self.monitor)
        if current_value is not None and current_value >= self.min_val_accuracy:
            super().on_validation_epoch_end(trainer, pl_module)
        else:
            # Reset wait count if below threshold
            self.wait_count = 0


def main():
    args, unknown = parser.parse_known_args()
    kwargs = parse_kwargs(unknown)

    # Initialize the logger
    logger = pl.loggers.WandbLogger(
        project=project_paths.project_name,
        save_dir=project_paths.logs,
        config=vars(args) | kwargs,
        tags=["train"],
    )
    # wandb.init()  # trick to log tables

    # Get the number of available CPU and GPU cores
    num_cpu_cores = multiprocessing.cpu_count()
    num_gpu_cores = torch.cuda.device_count()
    print(f"Number of available cores: CPU={num_cpu_cores}, GPU={num_gpu_cores}")

    # Load the dataset (using FFCV)
    dataloader_args = {
        "n_timesteps": args.n_timesteps,
        "batch_size": args.batch_size,
        "num_workers": 8,  # ToDo: outsource to config or find automatically
        "encoding": "image",
        "resolution": args.resolution,
    }

    train_loader = get_ffcv_dataloader(
        path=args.dataset_train,
        data_transform=f"{args.data_transform}_train_ffcv",
        **dataloader_args,
    )

    val_loader = get_ffcv_dataloader(
        path=args.dataset_val,
        data_transform=f"{args.data_transform}_test_ffcv",
        **dataloader_args,
    )

    # Load the dataset (using PyTorch)
    # dataset = get_dataset(
    #     args.dataset_train,
    #     data_transform=f"{args.data_transform}_train",
    #     target_transform=f"{args.data_transform}_all",
    #     )

    # Log example training image
    inputs, label_indices, *paths = next(train_loader.__iter__())
    inputs = _adjust_data_dimensions(inputs)
    label_indices = _adjust_label_dimensions(label_indices)

    print("Input Shape:", inputs.shape)
    print("Pixel Values:", inputs.mean(), "+-", inputs.std())

    batch_size, n_timesteps, *input_shape = inputs.shape
    logger.log_image(
        key="input_samples",
        images=[inputs[0, t] for t in range(n_timesteps)],
        caption=[label_indices[0, t] for t in range(n_timesteps)],
    )

    # Parse loss criterion
    criterion_params = [
        (loss, parse_string2dict(config))
        for loss, config in zip(args.loss, args.loss_config)
    ]

    # Load model state
    state_dict = torch.load(args.input_model_state)
    last_key = next(reversed(state_dict))
    n_classes = len(state_dict[last_key])

    # Load the model
    model = getattr(models, args.model_name)(
        input_dims=(n_timesteps, *input_shape),
        n_classes=n_classes,
        criterion_params=criterion_params,
        store_responses=args.store_responses,
        **kwargs,
    )

    # Apply the model state
    model.load_state_dict(state_dict)

    # CALLBACKS:
    callbacks = []

    ## custom callbacks for monitoring temporal dynamics
    if hasattr(model, "n_timesteps") and model.n_timesteps > 1:
        callbacks += [
            custom_callbacks.MonitorClassifierResponses(),
        ]
        if hasattr(model, "train_tau") and model.train_tau:
            callbacks += [
                custom_callbacks.MonitorTimescales(),
            ]

    callbacks += [
        custom_callbacks.MonitorWeightDistributions(),
    ]

    ## Define checkpoint callback
    checkpoint_path = project_paths.logs / "checkpoints" / args.output_model_state.stem

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",  # or any other metric you're tracking
        save_top_k=1,  # save only the best model
        mode="min",  # save the model with the minimum validation loss
        dirpath=checkpoint_path.parent,
        filename=checkpoint_path.name + "-{epoch:02d}-{val_loss:.2f}",
        save_last=True,  # save the last checkpoint
    )
    callbacks += [checkpoint_callback]

    ## Load the training checkpoint if it exists
    files = list(checkpoint_path.parent.glob(f"{checkpoint_path.name}*"))
    if len(files):
        checkpoint_path = files[-1]
        model = getattr(models, args.model_name).load_from_checkpoint(checkpoint_path)
        print(f"Loaded model from checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = None

    ## Define early stopping callback
    early_stop_callback = EarlyStoppingWithMin(
        monitor="val_accuracy", patience=5, mode="max", min_val_accuracy=0.7
    )
    callbacks += [early_stop_callback]

    ## Define learning rate monitor callback
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_intedvml="epoch")
    callbacks += [lr_monitor]

    ## Define device stats monitor callback
    # callbacks += [pl.callbacks.DeviceStatsMonitor()]

    # Setup the trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        max_epochs=args.epochs,
        logger=logger,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        precision=args.precision,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        accumulate_grad_batches=args.accumulate_grad_batches,
        profiler=None if args.profiler == "None" else args.profiler,
        enable_progress_bar=args.enable_progress_bar,
        benchmark=args.benchmark,
    )

    # Run training
    torch.set_float32_matmul_precision("medium")  # use medium precision to speed up
    torch.cuda.empty_cache()

    trainer.validate(model, val_loader)
    print("Training model...")
    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path=checkpoint_path,
    )

    # Save the trained model
    args.output_model_state.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_model_state)
    # wandb.save("model.pt")

    return None


if __name__ == "__main__":
    main()
