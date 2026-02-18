"""
Shared pytest fixtures and configuration for AlphaDeforest tests.
"""

import pytest
import torch
import numpy as np
import json
from typing import Dict, Any, List
from pathlib import Path


@pytest.fixture
def sample_dims() -> Dict[str, int]:
    """
    Standard dimensions for testing.
    
    Returns:
        Dict[str, int]: Dictionary with B (Batch), T (Time), D (Channels), H (Height), W (Width).
    """
    return {"B": 2, "T": 3, "D": 64, "H": 128, "W": 128}


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """
    Minimum valid configuration based on Pydantic schemas.
    
    Returns:
        Dict[str, Any]: Mocked configuration dictionary.
    """
    return {
        "model": {
            "embedding_dim": 64, 
            "latent_dim": 32, 
            "cae_h": 16, 
            "cae_w": 16, 
            "hidden_dim_mem": 128
        },
        "data": {
            "dataset_dir": "data/", 
            "batch_size": 2, 
            "num_workers": 0,
            "shuffle": True, 
            "train_years": [2020, 2021], 
            "test_year": [2022]
        },
        "train": {
            "epochs": 1, 
            "lr": 1e-3, 
            "device": "cpu", 
            "use_amp": False,
            "lambda_rec": 1.0,
            "lambda_pred": 0.5,
            "seed": 42,
            "gamma": 0.94
        },
        "experiment": {
            "name": "test_exp", 
            "version": 1, 
            "output_dir": "outputs/"
        }
    }


@pytest.fixture
def input_tensor(sample_dims: Dict[str, int]) -> torch.Tensor:
    """
    Generates a synthetic input tensor of shape (B, T, D, H, W).

    Args:
        sample_dims (Dict[str, int]): Dimensions for the tensor.

    Returns:
        torch.Tensor: Randomized tensor.
    """
    d = sample_dims
    return torch.randn(d["B"], d["T"], d["D"], d["H"], d["W"])


@pytest.fixture
def mock_dataset_dir(tmp_path: Path) -> str:
    """
    Creates a temporary directory with .json and .npy files for dataset testing.

    Args:
        tmp_path (Path): Pytest fixture for a temporary path.

    Returns:
        str: Path to the mocked dataset directory.
    """
    d = tmp_path / "mock_data"
    d.mkdir()
    
    # Create 2 years for 1 tile to form a sequence
    tile_id = "tile_001"
    for year in [2020, 2021]:
        base_name = f"{tile_id}_{year}"
        # Metadata
        meta = {"tile_id": tile_id, "year": year, "row": 0, "col": 0}
        with open(d / f"{base_name}.meta.json", "w") as f:
            json.dump(meta, f)
        # Embedding (H, W, D) as expected by dataset.py before permuting to (D, H, W)
        emb = np.random.randn(128, 128, 64).astype(np.float32)
        np.save(d / f"{base_name}.emb.npy", emb)
        
    return str(d)