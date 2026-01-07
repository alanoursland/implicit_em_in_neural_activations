import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import sys


class GPUCachedDataLoader:
    """Dataloader that preloads entire dataset to GPU for fast iteration."""
    def __init__(self, dataset, batch_size, shuffle=True, device='cuda'):
        # Load all data to GPU at once.
        # Accept either a standard dataset (iterable yielding (x, y)) or a (data, labels) tuple.
        if isinstance(dataset, tuple) and len(dataset) == 2:
            data, labels = dataset
            self.data = data.to(device)
            self.labels = labels.to(device)
        else:
            all_data = []
            all_labels = []
            for x, y in dataset:
                all_data.append(x)
                all_labels.append(y)

            self.data = torch.stack(all_data).to(device)
            self.labels = torch.tensor(all_labels).to(device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

    def __iter__(self):
        n = len(self.data)
        if self.shuffle:
            indices = torch.randperm(n, device=self.device)
        else:
            indices = torch.arange(n, device=self.device)

        for i in range(0, n, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.data[batch_indices], self.labels[batch_indices]

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size

# Custom transform that can be pickled
class FlattenTransform:
    def __call__(self, x):
        return x.view(-1)

def get_mnist(batch_size: int, num_workers: int = 4, use_gpu_cache: bool = True):
    """
    Load MNIST, flatten to vectors.

    Args:
        batch_size: Batch size for data loading. Use -1 for full dataset (60000 train, 10000 test)
        num_workers: Number of worker processes (set to 0 on Windows)
        use_gpu_cache: If True and CUDA is available, preload entire dataset to GPU
    """
    # Use num_workers=0 on Windows to avoid pickling issues
    if sys.platform == 'win32':
        num_workers = 0
        print(f"    Windows detected, using num_workers=0")

    transform = transforms.Compose([transforms.ToTensor(), FlattenTransform()])  # Flatten to 784

    print(f"    Downloading/loading MNIST dataset...")
    train_dataset = datasets.MNIST(
        root=r"E:\ml_datasets", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=r"E:\ml_datasets", train=False, download=True, transform=transform
    )
    print(f"    Train samples: {len(train_dataset):,}")
    print(f"    Test samples: {len(test_dataset):,}")

    # Handle full batch mode
    if batch_size == -1:
        batch_size = len(train_dataset)
        print(f"    Using full batch mode: batch_size = {batch_size}")

    # Use GPU caching if available and requested
    if use_gpu_cache and torch.cuda.is_available():
        print(f"    Caching dataset to GPU memory...")
        # Avoid per-sample Python iteration (very slow) by using MNIST's underlying tensors.
        # train_dataset.data: uint8 (N, 28, 28); train_dataset.targets: int64 (N,)
        train_x = train_dataset.data.float().div_(255.0).view(len(train_dataset), -1)
        train_y = train_dataset.targets
        test_x = test_dataset.data.float().div_(255.0).view(len(test_dataset), -1)
        test_y = test_dataset.targets

        train_loader = GPUCachedDataLoader((train_x, train_y), batch_size=batch_size, shuffle=True, device="cuda")
        test_loader = GPUCachedDataLoader((test_x, test_y), batch_size=len(test_dataset), shuffle=False, device="cuda")
        print(f"    ✓ Dataset cached to GPU")
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    return train_loader, test_loader, 784

def get_synthetic_2d(batch_size: int, n_samples: int = 10000, n_clusters: int = 8):
    """
    Generate 2D Gaussian clusters.
    """
    import numpy as np

    np.random.seed(42)

    # Cluster centers on a circle
    angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
    centers = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 2

    # Generate samples
    samples_per_cluster = n_samples // n_clusters
    X = []
    y = []

    for i, center in enumerate(centers):
        samples = np.random.randn(samples_per_cluster, 2) * 0.3 + center
        X.append(samples)
        y.extend([i] * samples_per_cluster)

    X = np.concatenate(X, axis=0).astype(np.float32)
    y = np.array(y)

    # Shuffle
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]

    # Split
    split = int(0.8 * len(X))
    train_dataset = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(y[:split]))
    test_dataset = TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(y[split:]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, 2

def get_data(name: str, batch_size: int, **kwargs):
    """Factory function."""
    if name == "mnist":
        return get_mnist(batch_size, **kwargs)
    elif name == "synthetic_2d":
        return get_synthetic_2d(batch_size, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")
