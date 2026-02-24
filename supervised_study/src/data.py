"""Data loading for supervised implicit EM study."""

import torch
from torch.utils.data import DataLoader
from typing import Tuple


class GPUCachedDataLoader:
    """DataLoader that preloads entire dataset to GPU for fast iteration."""

    def __init__(self, data, labels, batch_size, shuffle=True, device="cuda"):
        self.data = data.to(device)
        self.labels = labels.to(device)
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
            batch_indices = indices[i : i + self.batch_size]
            yield self.data[batch_indices], self.labels[batch_indices]

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size


def get_mnist(
    batch_size: int = 128,
    flatten: bool = True,
    data_dir: str = "E:/ml_datasets",
    use_gpu_cache: bool = True,
    device: str = "cuda",
) -> Tuple[DataLoader, DataLoader]:
    """Load MNIST dataset.

    Args:
        batch_size: Batch size
        flatten: Flatten images to 784-d vectors
        data_dir: Directory to download/load data
        use_gpu_cache: Preload dataset to GPU if available
        device: Device to cache data on

    Returns:
        (train_loader, test_loader)
    """
    from torchvision import datasets, transforms

    transform_list = [transforms.ToTensor()]
    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))
    transform = transforms.Compose(transform_list)

    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    if use_gpu_cache and torch.cuda.is_available():
        train_x = train_dataset.data.float().div_(255.0)
        train_y = train_dataset.targets
        test_x = test_dataset.data.float().div_(255.0)
        test_y = test_dataset.targets

        if flatten:
            train_x = train_x.view(len(train_dataset), -1)
            test_x = test_x.view(len(test_dataset), -1)

        train_loader = GPUCachedDataLoader(train_x, train_y, batch_size, shuffle=True, device=device)
        test_loader = GPUCachedDataLoader(test_x, test_y, batch_size, shuffle=False, device=device)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, test_loader
