# debug_ddp.py
import os
import sys
import torch
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import logging

sys.path.append("/home/rg5022/DynVision")

from dynvision import models
from dynvision.data.ffcv_dataloader import get_ffcv_dataloader
from dynvision.runtime.train_model_new import setup_callbacks
from dynvision.project_paths import project_paths

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def debug_ddp_environment():
    """Debug DDP environment and setup."""
    print("=== DDP Environment Debug ===")

    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Check environment variables
    env_vars = [
        "CUDA_VISIBLE_DEVICES",
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "NODE_RANK",
    ]
    print("\nEnvironment Variables:")
    for var in env_vars:
        print(f"  {var}: {os.environ.get(var, 'Not set')}")

    # Check PyTorch and Lightning versions
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"PyTorch Lightning version: {pl.__version__}")

    # Check if distributed is available
    print(f"Distributed available: {dist.is_available()}")
    if dist.is_available():
        print(f"NCCL available: {dist.is_nccl_available()}")
        print(f"MPI available: {dist.is_mpi_available()}")
        print(f"Gloo available: {dist.is_gloo_available()}")


def check_dtype_consistency(model, sample_input):
    """Check dtype consistency between model and input."""
    print("\n=== Dtype Consistency Check ===")

    try:
        model_dtype = next(model.parameters()).dtype
        model_device = next(model.parameters()).device
        print(f"Model parameters - dtype: {model_dtype}, device: {model_device}")
    except StopIteration:
        print("Warning: Model has no parameters")
        return False

    print(f"Sample input - dtype: {sample_input.dtype}, device: {sample_input.device}")

    # Check consistency
    dtype_match = model_dtype == sample_input.dtype
    device_compatible = (model_device.type == sample_input.device.type) or (
        sample_input.device.type == "cpu" and model_device.type == "cuda"
    )

    print(f"Dtype match: {dtype_match}")
    print(f"Device compatible: {device_compatible}")

    if not dtype_match:
        print(f"⚠️  Dtype mismatch detected!")
        print(f"   Model: {model_dtype}, Input: {sample_input.dtype}")
        print("   Suggestion: Ensure input and model have same dtype")

    if not device_compatible:
        print(f"⚠️  Device incompatibility detected!")
        print(f"   Model: {model_device}, Input: {sample_input.device}")
        print("   Suggestion: Move input to same device as model")

    return dtype_match and device_compatible


def test_minimal_ddp():
    """Test minimal DDP setup."""
    print("\n=== Testing Minimal DDP Setup ===")

    # Determine target device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # Start with float32 as default

    print(f"Using device: {device}, dtype: {dtype}")

    # Create minimal model
    model = models.DyRCNNx4(
        input_dims=(20, 3, 32, 32),
        n_classes=10,
        n_timesteps=20,
        recurrence_type="self",  # Use simpler recurrence for testing
    )

    # Move model to device and ensure consistent dtype
    model = model.to(device=device, dtype=dtype)

    # Get actual dtype from model parameters after moving to device
    try:
        actual_dtype = next(model.parameters()).dtype
        actual_device = next(model.parameters()).device
        print(f"Model parameters - device: {actual_device}, dtype: {actual_dtype}")

        # Ensure model parameters have consistent dtype
        if hasattr(model, "_ensure_parameter_dtypes"):
            model._ensure_parameter_dtypes(target_dtype=actual_dtype)
        else:
            print("Warning: Model doesn't have _ensure_parameter_dtypes method")

    except StopIteration:
        print("Warning: Model has no parameters")
        actual_dtype = dtype
        actual_device = device

    # Test model forward pass with matching dtype and device
    print("Testing model forward pass...")
    try:
        # Create dummy input with same dtype and device as model
        dummy_input = torch.randn(
            2, 20, 3, 32, 32, dtype=actual_dtype, device=actual_device
        )
        print(
            f"Dummy input - device: {dummy_input.device}, dtype: {dummy_input.dtype}"
        )

        # Test forward pass
        with torch.no_grad():  # Avoid building computation graph for testing
            output = model(dummy_input)
        print(f"✅ Model forward pass successful. Output shape: {output.shape}")
        print(f"Output - device: {output.device}, dtype: {output.dtype}")

    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test trainer creation with DDP
    print("Testing trainer creation with DDP...")
    try:
        strategy = DDPStrategy(
            process_group_backend="nccl" if torch.cuda.is_available() else "gloo",
            find_unused_parameters=True,  # Add this for debugging
            static_graph=False,
        )

        trainer = pl.Trainer(
            strategy=strategy,
            devices=(
                min(2, torch.cuda.device_count()) if torch.cuda.is_available() else 1
            ),
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=2,
            enable_checkpointing=False,  # Disable for testing
            logger=False,  # Disable logging for testing
            enable_progress_bar=False,
            precision=32,  # Explicitly set precision to match our dtype
        )
        print("✅ Trainer creation successful")
        print(f"Trainer precision: {trainer.precision}")
        print(f"Trainer strategy: {trainer.strategy}")

        return trainer, model
    except Exception as e:
        print(f"❌ Trainer creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_dataloader():
    """Test dataloader creation."""
    print("\n=== Testing DataLoader ===")
    try:
        # Determine appropriate dtype and device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32

        # Create a simple dataloader for testing with consistent dtype
        dummy_data = torch.randn(32, 3, 32, 32, dtype=dtype)
        dummy_labels = torch.randint(0, 10, (32,))

        print(f"Dummy data - device: {dummy_data.device}, dtype: {dummy_data.dtype}")
        print(
            f"Dummy labels - device: {dummy_labels.device}, dtype: {dummy_labels.dtype}"
        )

        dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # Use 0 workers for testing
            pin_memory=False,
        )

        # Test loading a batch
        batch = next(iter(dataloader))
        batch_data, batch_labels = batch
        print(f"✅ DataLoader test successful.")
        print(
            f"Batch data shape: {batch_data.shape}, dtype: {batch_data.dtype}, device: {batch_data.device}"
        )
        print(
            f"Batch labels shape: {batch_labels.shape}, dtype: {batch_labels.dtype}, device: {batch_labels.device}"
        )

        return dataloader
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def run_minimal_ddp_test():
    """Run minimal DDP training test."""
    print("\n=== Running Minimal DDP Test ===")

    # Debug environment
    debug_ddp_environment()

    # Test dataloader
    train_loader = test_dataloader()
    if train_loader is None:
        return

    # Test DDP setup
    result = test_minimal_ddp()
    if result is False:
        return

    trainer, model = result

    # Check dtype consistency with actual data from loader
    print("\n=== Checking Dtype Consistency with DataLoader ===")
    try:
        sample_batch = next(iter(train_loader))
        sample_data, sample_labels = sample_batch

        # Move sample to same device as model for testing
        model_device = next(model.parameters()).device
        sample_data = sample_data.to(model_device)

        dtype_consistent = check_dtype_consistency(model, sample_data)

        if not dtype_consistent:
            print("⚠️  Fixing dtype/device inconsistencies...")
            # Ensure model parameters match expected dtype
            if hasattr(model, "_ensure_parameter_dtypes"):
                model._ensure_parameter_dtypes(target_dtype=torch.float32)

    except Exception as e:
        print(f"❌ Dtype consistency check failed: {e}")

    # Try fitting
    print("Attempting DDP training...")
    try:
        trainer.fit(model, train_loader, train_loader)
        print("✅ DDP training test successful!")
    except Exception as e:
        print(f"❌ DDP training failed: {e}")
        import traceback

        traceback.print_exc()

        # Try alternative strategies
        print("\n=== Trying Alternative Strategies ===")
        test_alternatives(model, train_loader)


def test_alternatives(model, train_loader):
    """Test alternative training strategies."""

    # Test 1: Single GPU
    print("Testing single GPU...")
    try:
        trainer = pl.Trainer(
            devices=1,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            max_epochs=1,
            limit_train_batches=2,
            enable_checkpointing=False,
            logger=False,
        )
        trainer.fit(model, train_loader, train_loader)
        print("✅ Single GPU training successful")
    except Exception as e:
        print(f"❌ Single GPU training failed: {e}")

    # Test 2: DataParallel (DP) instead of DDP
    if torch.cuda.device_count() > 1:
        print("Testing DataParallel (DP)...")
        try:
            trainer = pl.Trainer(
                strategy="dp",  # Use DataParallel instead of DDP
                devices=min(2, torch.cuda.device_count()),
                accelerator="gpu",
                max_epochs=1,
                limit_train_batches=2,
                enable_checkpointing=False,
                logger=False,
            )
            trainer.fit(model, train_loader, train_loader)
            print("✅ DataParallel training successful")
        except Exception as e:
            print(f"❌ DataParallel training failed: {e}")


if __name__ == "__main__":
    run_minimal_ddp_test()
