"""
Unit tests for data loading and dataset building in AlphaDeforest.
"""

import torch
from alphadeforest.data.dataset import AlphaEarthTemporalDataset, get_dataloader


def test_dataset_sequence_building(mock_dataset_dir: str) -> None:
    """
    Verifies that the dataset correctly pairs JSON and NPY files into temporal sequences.

    Args:
        mock_dataset_dir (str): Path to the temporary mock data provided by conftest.
    """
    # Only include years created in the mock
    dataset = AlphaEarthTemporalDataset(
        dataset_dir=mock_dataset_dir,
        years=[2020, 2021],
        mode="train"
    )
    
    assert len(dataset) == 1, "Should have detected exactly one sequence (tile_001)"
    
    seq = dataset[0]
    # Expected shape: (T, D, H, W) -> (2, 64, 128, 128)
    assert seq.shape[0] == 2, "Sequence must have 2 time steps"
    assert seq.shape[1] == 64, "Incorrect number of embedding channels"
    assert isinstance(seq, torch.Tensor), "Output should be a PyTorch tensor"


def test_dataloader_batching(mock_dataset_dir: str) -> None:
    """
    Verifies that the DataLoader correctly batches sequences.

    Args:
        mock_dataset_dir (str): Path to the temporary mock data provided by conftest.
    """
    dataset = AlphaEarthTemporalDataset(mock_dataset_dir, years=[2020, 2021])
    # Force batch_size=1 since there is only one sequence in the mock
    loader = get_dataloader(dataset, batch_size=1, num_workers=0)
    
    batch = next(iter(loader))
    assert batch.ndim == 5, "Batch should have shape (B, T, D, H, W) with B=1"