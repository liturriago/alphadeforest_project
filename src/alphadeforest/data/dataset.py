"""
Dataset module for AlphaDeforest.
This module handles loading temporal sequences of embeddings for deforestation detection.
"""

import os
import json
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional


class AlphaEarthTemporalDataset(Dataset):
    """
    Dataset for loading temporal sequences of AlphaEarth embeddings.
    Assumes tile_id is a stable spatial identifier (rX_cY).
    """

    def __init__(
        self,
        dataset_dir: str,
        years: List[int],
        mode: str = "train",
        transform: Optional[Any] = None,
    ):
        assert mode in ["train", "full"]

        self.dataset_dir = dataset_dir
        self.years = sorted(years)   # ← deterministic ordering
        self.years_set = set(years)
        self.mode = mode
        self.transform = transform

        self.tile_metadata: List[Dict[str, Any]] = []
        self.sequences = self._load_sequences()

    def _load_sequences(self) -> List[np.ndarray]:

        tiles = defaultdict(dict)   # ← IMPORTANT CHANGE

        files = os.listdir(self.dataset_dir)
        json_files = sorted(f for f in files if f.endswith(".meta.json"))

        if len(json_files) == 0:
            raise RuntimeError("No .meta.json files found.")

        print(f"Detected metadata files: {len(json_files)}")

        # ------------------------------------------------------
        # Load metadata + embeddings
        # ------------------------------------------------------
        for json_name in json_files:

            json_path = os.path.join(self.dataset_dir, json_name)

            with open(json_path, "r") as f:
                meta = json.load(f)

            npy_name = json_name.replace(".meta.json", ".emb.npy")
            npy_path = os.path.join(self.dataset_dir, npy_name)

            if not os.path.exists(npy_path):
                continue

            emb = np.load(npy_path)

            if emb.ndim != 3:
                print(f"Invalid embedding: {npy_name} -> shape {emb.shape}")
                continue

            tile_id = meta["tile_id"]
            year = meta["year"]

            if year not in self.years_set:
                continue

            # Store by year (prevents duplicates / disorder)
            tiles[tile_id][year] = emb

        # ------------------------------------------------------
        # Build temporal sequences
        # ------------------------------------------------------
        sequences = []

        for tile_id in sorted(tiles.keys()):

            year_map = tiles[tile_id]

            # Keep only available years (ordered)
            available_years = sorted(year_map.keys())

            if len(available_years) < 2:
                continue

            seq = np.stack(
                [year_map[y] for y in available_years],
                axis=0
            )

            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

            sequences.append(seq)

            # Extract row/col from tile_id (cleaner & safer)
            row, col = self._parse_tile_id(tile_id)

            self.tile_metadata.append({
                "row": row,
                "col": col,
                "tile_id": tile_id,
                "years": available_years
            })

        print(f"Valid sequences built: {len(sequences)}")

        if len(sequences) == 0:
            raise RuntimeError(
                "No sequences were built. "
                "Check year filtering or dataset consistency."
            )

        return sequences

    @staticmethod
    def _parse_tile_id(tile_id: str):
        """
        Parse tile_id = rX_cY → (X, Y)
        """
        try:
            parts = tile_id.split("_")
            row = int(parts[0][1:])
            col = int(parts[1][1:])
            return row, col
        except Exception:
            return 0, 0

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:

        seq = self.sequences[idx]

        # (T, H, W, D) → (T, D, H, W)
        seq = torch.from_numpy(seq).float().permute(0, 3, 1, 2)

        if self.transform:
            seq = self.transform(seq)

        return seq

def get_dataloader(
    dataset: Dataset, 
    batch_size: int, 
    num_workers: int, 
    partition: str = "train",
    pin_memory: bool = True
) -> DataLoader:
    """
    Creates a DataLoader for the given dataset.

    Args:
        dataset (Dataset): The dataset to load.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads for data loading.
        partition (str): Dataset partition ("train" or other). Defaults to "train".
        pin_memory (bool): Whether to pin memory for faster GPU transfer. Defaults to True.

    Returns:
        DataLoader: The configured DataLoader.
    """
    is_train = partition.lower() == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=is_train and len(dataset) >= batch_size
    )