import importlib.util
from pathlib import Path

# Get the current directory
current_dir = Path(__file__).resolve().parent

# Get all Python files that start with 'data_'
files = current_dir.glob("data_*.py")

# Import all classes and functions from the files
for file in files:
    module_name = file.stem

    # Create a module spec
    spec = importlib.util.spec_from_file_location(module_name, file)

    # Create a module from the spec
    module = importlib.util.module_from_spec(spec)

    # Execute the module
    spec.loader.exec_module(module)

    # Add the classes and functions to the __all__ list
    globals().update(
        {
            k: v
            for k, v in module.__dict__.items()  # noqa: E203
            if not k.startswith("_")
        }
    )
