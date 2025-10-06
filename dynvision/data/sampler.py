import torch
from torch.utils.data import DataLoader, Sampler
from collections import defaultdict
import random


class RoundRobinSampler(Sampler):
    """
    A sampler that loads data in round-robin fashion across categories.
    This ensures that early batches contain representative samples from all categories.

    Example order for 3 categories with 3 samples each:
    Instead of: [0,0,0,1,1,1,2,2,2]
    Produces: [0,1,2,0,1,2,0,1,2]
    """

    def __init__(
        self, dataset, get_label_fn=None, shuffle_within_class=False, seed=None
    ):
        """
        Args:
            dataset: The dataset to sample from
            get_label_fn: Function to extract label from dataset[idx].
                         If None, assumes dataset[idx][1] is the label
            shuffle_within_class: Whether to shuffle indices within each class
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.shuffle_within_class = shuffle_within_class
        self.seed = seed

        if get_label_fn is None:
            # Default: assume dataset returns (data, label) tuples
            def default_get_label(dataset, idx):
                return dataset[idx][1]

            self.get_label_fn = default_get_label
        else:
            self.get_label_fn = get_label_fn

        # Group indices by class
        self.class_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            label = self.get_label_fn(dataset, idx)
            # Convert tensor labels to int if needed
            if hasattr(label, "item"):
                label = label.item()
            self.class_to_indices[label].append(idx)

        self.classes = sorted(self.class_to_indices.keys())
        self.num_classes = len(self.classes)
        self.length = len(dataset)

        print(f"RoundRobinSampler initialized:")
        print(f"  Classes: {self.classes}")
        print(
            f"  Samples per class: {[len(self.class_to_indices[c]) for c in self.classes]}"
        )

    def __iter__(self):
        if self.seed is not None:
            random.seed(self.seed)

        # Prepare iterators for each class
        class_iterators = {}
        for class_label in self.classes:
            indices = self.class_to_indices[class_label].copy()
            if self.shuffle_within_class:
                random.shuffle(indices)
            class_iterators[class_label] = iter(indices)

        # Round-robin through classes
        result = []
        active_classes = list(self.classes)

        while active_classes:
            for class_label in active_classes[
                :
            ]:  # Use slice copy to avoid modification during iteration
                try:
                    idx = next(class_iterators[class_label])
                    result.append(idx)
                except StopIteration:
                    # This class is exhausted, remove it from active classes
                    active_classes.remove(class_label)

        return iter(result)

    def __len__(self):
        return self.length


if __name__ == "__main__":
    # Run examples
    print("=== Round-Robin Sampler Examples ===")

    # Simple demonstration without external datasets
    print("\nSimple demonstration:")

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self):
            # Create samples: 3 samples each for classes 0, 1, 2
            self.data = [(i, i // 3) for i in range(9)]  # (sample_id, class)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sample_id, class_label = self.data[idx]
            return torch.tensor([sample_id]), torch.tensor(class_label)

    simple_dataset = SimpleDataset()
    simple_sampler = RoundRobinSampler(simple_dataset, seed=42)
    simple_dataloader = DataLoader(
        simple_dataset, batch_size=9, sampler=simple_sampler
    )

    for data, labels in simple_dataloader:
        print(f"Round-robin order - Sample IDs: {data.flatten().tolist()}")
        print(f"Round-robin order - Class labels: {labels.tolist()}")
        break  # Only need first batch for demonstration

    # Compare with regular sequential loading
    regular_dataloader = DataLoader(simple_dataset, batch_size=9, shuffle=False)
    for data, labels in regular_dataloader:
        print(f"Sequential order - Sample IDs: {data.flatten().tolist()}")
        print(f"Sequential order - Class labels: {labels.tolist()}")
        break
