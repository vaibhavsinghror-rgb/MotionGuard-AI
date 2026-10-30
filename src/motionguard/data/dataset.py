"""PyTorch Dataset and DataLoaders for MotionGuard AI."""

from typing import Callable, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class HARDataset(Dataset):
    """PyTorch Dataset for 6-channel Human Activity Recognition time series."""

    def __init__(
        self,
        waveforms: np.ndarray,
        labels: Optional[np.ndarray] = None,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        """
        Args:
            waveforms: np.ndarray of shape (N, 6, 300)
            labels: optional np.ndarray of shape (N,)
            transform: optional callable augmentation function
        """
        assert waveforms.ndim == 3, f"Expected waveforms shape (N, C, T), got {waveforms.shape}"
        self.waveforms = torch.from_numpy(waveforms).float()
        self.labels = torch.from_numpy(labels).long() if labels is not None else None
        self.transform = transform

    def __len__(self) -> int:
        return len(self.waveforms)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        x = self.waveforms[idx]  # shape: (6, 300)
        
        if self.transform is not None:
            # apply augmentation on numpy representation if desired
            x_np = x.numpy()
            x_np = self.transform(x_np)
            x = torch.from_numpy(x_np).float()

        if self.labels is not None:
            y = self.labels[idx]
            return x, y
        return (x,)


def jitter_and_scale(
    x: np.ndarray,
    sigma: float = 0.05,
    scale_range: Tuple[float, float] = (0.95, 1.05),
) -> np.ndarray:
    """Applies Gaussian noise jitter and random scaling for time-series augmentation."""
    noise = np.random.normal(loc=0.0, scale=sigma, size=x.shape).astype(np.float32)
    factor = np.random.uniform(scale_range[0], scale_range[1])
    return (x + noise) * factor


def create_dataloaders(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: Optional[np.ndarray] = None,
    test_y: Optional[np.ndarray] = None,
    batch_size: int = 64,
    augment_train: bool = False,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Creates standard PyTorch DataLoaders for train, val, and test splits."""
    train_transform = jitter_and_scale if augment_train else None
    
    train_ds = HARDataset(train_x, train_y, transform=train_transform)
    val_ds = HARDataset(val_x, val_y)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = None
    if test_x is not None and test_y is not None:
        test_ds = HARDataset(test_x, test_y)
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader, test_loader
