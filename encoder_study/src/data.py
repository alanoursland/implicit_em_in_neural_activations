"""Data loading utilities for encoder study."""

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional
from pathlib import Path


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
        batch_size: Batch size for data loaders
        flatten: If True, flatten images to 784-dimensional vectors
        data_dir: Directory to download/load data
        use_gpu_cache: If True and CUDA available, preload dataset to GPU
        device: Device to cache data on (default: "cuda")

    Returns:
        train_loader: Training data loader
        test_loader: Test data loader
    """
    from torchvision import datasets, transforms

    transform_list = [transforms.ToTensor()]
    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))

    transform = transforms.Compose(transform_list)

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    # Use GPU caching if available and requested
    if use_gpu_cache and torch.cuda.is_available():
        # Load directly from underlying tensors (much faster than iterating)
        train_x = train_dataset.data.float().div_(255.0)
        train_y = train_dataset.targets
        test_x = test_dataset.data.float().div_(255.0)
        test_y = test_dataset.targets

        if flatten:
            train_x = train_x.view(len(train_dataset), -1)
            test_x = test_x.view(len(test_dataset), -1)

        train_loader = GPUCachedDataLoader(
            train_x, train_y, batch_size=batch_size, shuffle=True, device=device
        )
        test_loader = GPUCachedDataLoader(
            test_x, test_y, batch_size=batch_size, shuffle=False, device=device
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    return train_loader, test_loader


def get_synthetic(
    n_samples: int = 10000,
    n_features: int = 784,
    n_components: int = 10,
    noise_std: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic data with known ground truth components.

    Data is generated as a mixture: x = sum_j(z_j * c_j) + noise
    where c_j are component vectors and z_j are sparse coefficients.

    Args:
        n_samples: Number of samples to generate
        n_features: Dimensionality of each sample
        n_components: Number of true components
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        data: Tensor of shape (n_samples, n_features)
        true_components: Tensor of shape (n_components, n_features)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Generate random orthogonal components
    components = torch.randn(n_components, n_features)
    components = components / components.norm(dim=1, keepdim=True)

    # Generate sparse coefficients (each sample uses 1-3 components)
    coefficients = torch.zeros(n_samples, n_components)
    for i in range(n_samples):
        n_active = np.random.randint(1, min(4, n_components + 1))
        active_idx = np.random.choice(n_components, n_active, replace=False)
        coefficients[i, active_idx] = torch.rand(n_active) + 0.5

    # Generate data
    data = coefficients @ components
    data = data + noise_std * torch.randn_like(data)

    return data, components


def get_synthetic_loader(
    n_samples: int = 10000,
    n_features: int = 784,
    n_components: int = 10,
    batch_size: int = 128,
    noise_std: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, torch.Tensor]:
    """Get synthetic data as a DataLoader.

    Args:
        n_samples: Number of samples to generate
        n_features: Dimensionality of each sample
        n_components: Number of true components
        batch_size: Batch size for data loader
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        loader: DataLoader for synthetic data
        true_components: Tensor of shape (n_components, n_features)
    """
    data, components = get_synthetic(
        n_samples=n_samples,
        n_features=n_features,
        n_components=n_components,
        noise_std=noise_std,
        seed=seed,
    )

    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader, components


def get_llm_activations(
    model_name: str = "gpt2",
    layer: int = 6,
    cache_path: str = "./data/llm_activations",
    batch_size: int = 128,
    max_samples: int = 50000,
) -> Tuple[DataLoader, DataLoader]:
    """Load cached LLM activations or generate them.

    Args:
        model_name: Name of the transformer model
        layer: Layer to extract activations from
        cache_path: Path to cache activations
        batch_size: Batch size for data loaders
        max_samples: Maximum number of samples to use

    Returns:
        train_loader: Training data loader
        test_loader: Test data loader
    """
    cache_dir = Path(cache_path)
    cache_file = cache_dir / f"{model_name}_layer{layer}_activations.pt"

    if cache_file.exists():
        # Load cached activations
        data = torch.load(cache_file)
        train_data = data["train"]
        test_data = data["test"]
    else:
        # Generate activations (requires transformers and datasets)
        train_data, test_data = _extract_llm_activations(
            model_name=model_name,
            layer=layer,
            max_samples=max_samples,
        )

        # Cache for future use
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"train": train_data, "test": test_data}, cache_file)

    train_dataset = TensorDataset(train_data)
    test_dataset = TensorDataset(test_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader


def _extract_llm_activations(
    model_name: str,
    layer: int,
    max_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract activations from a transformer model.

    Requires: transformers, datasets

    Args:
        model_name: Name of the transformer model
        layer: Layer to extract activations from
        max_samples: Maximum number of samples

    Returns:
        train_activations: Tensor of training activations
        test_activations: Tensor of test activations
    """
    try:
        from transformers import AutoModel, AutoTokenizer
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "LLM activation extraction requires 'transformers' and 'datasets'. "
            "Install with: pip install transformers datasets"
        )

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    activations = []
    hook_output = []

    def hook_fn(module, input, output):
        # For GPT-2, output is a tuple; we want the hidden states
        if isinstance(output, tuple):
            hook_output.append(output[0].detach())
        else:
            hook_output.append(output.detach())

    # Register hook on the specified layer
    if "gpt2" in model_name.lower():
        target_layer = model.h[layer]
    else:
        # Generic transformer - try common patterns
        if hasattr(model, "encoder"):
            target_layer = model.encoder.layer[layer]
        elif hasattr(model, "layers"):
            target_layer = model.layers[layer]
        else:
            raise ValueError(f"Cannot find layer {layer} in model {model_name}")

    handle = target_layer.register_forward_hook(hook_fn)

    # Extract activations
    n_collected = 0
    with torch.no_grad():
        for split in ["train", "test"]:
            split_activations = []
            for example in dataset[split]:
                if not example["text"].strip():
                    continue

                tokens = tokenizer(
                    example["text"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )

                hook_output.clear()
                model(**tokens)

                if hook_output:
                    # Take mean over sequence length
                    act = hook_output[0].mean(dim=1)  # (1, hidden_dim)
                    split_activations.append(act)
                    n_collected += act.shape[0]

                if n_collected >= max_samples:
                    break

            activations.append(torch.cat(split_activations, dim=0))

    handle.remove()

    # Split into train/test (80/20)
    all_activations = torch.cat(activations, dim=0)
    n_train = int(0.8 * len(all_activations))

    return all_activations[:n_train], all_activations[n_train:]
