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
from typing import List, Dict, Any, Optional, Set, Tuple


class AlphaEarthTemporalDataset(Dataset):
    """
    Dataset class for loading temporal sequences of satellite image embeddings.
    """
    def __init__(
        self,
        dataset_dir: str,
        years: List[int],
        mode: str = "train",
        transform: Optional[Any] = None,
    ):
        """
        Initializes the dataset.

        Args:
            dataset_dir (str): Directory where JSON and NPY files are located.
            years (List[int]): List of years to include in the dataset.
            mode (str): Dataset mode, either "train" or "full". Defaults to "train".
            transform (Optional[Any]): Optional transform to apply to the sequences.
        """
        assert mode in ["train", "full"]

        self.dataset_dir = dataset_dir
        self.years = set(years)
        self.mode = mode
        self.transform = transform

        self.tile_metadata: List[Dict[str, Any]] = []
        self.sequences = self._load_sequences()

    def _load_sequences(self) -> List[np.ndarray]:
        """
        Loads sequences from the dataset directory.
        Identifies JSON/NPY pairs and builds temporal sequences for each tile.

        Returns:
            List[np.ndarray]: List of temporal sequences.
        """
        tiles = defaultdict(list)
        coords_map = {}

        files = os.listdir(self.dataset_dir)
        json_files = [f for f in files if f.endswith(".json")]

        if len(json_files) == 0:
            raise RuntimeError("No JSON files found in the dataset directory.")

        print(f"Detected JSON files: {len(json_files)}")

        # Load metadata + embeddings
        for json_name in json_files:
            json_path = os.path.join(self.dataset_dir, json_name)

            with open(json_path, "r") as f:
                meta = json.load(f)

            npy_name = json_name.replace(".json", ".npy")
            npy_path = os.path.join(self.dataset_dir, npy_name)

            if not os.path.exists(npy_path):
                continue

            emb = np.load(npy_path)

            # Critical defensive validation
            if emb.ndim != 3:
                print(f"Invalid embedding: {npy_name} -> shape {emb.shape}")
                continue

            tile_id = meta["tile_id"]
            year = meta["year"]

            if year not in self.years:
                continue

            tiles[tile_id].append({
                "year": year,
                "emb": emb
            })

            if tile_id not in coords_map:
                coords_map[tile_id] = {
                    "row": meta.get("row", 0),
                    "col": meta.get("col", 0),
                    "tile_id": tile_id
                }

        # Build temporal sequences
        sequences = []

        for tile_id in sorted(tiles.keys()):
            samples = sorted(tiles[tile_id], key=lambda x: x["year"])

            if len(samples) < 2:
                continue

            seq = np.stack([s["emb"] for s in samples], axis=0)

            # Numerical cleaning (REALLY important)
            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

            sequences.append(seq)

            self.tile_metadata.append({
                "row": coords_map[tile_id]["row"],
                "col": coords_map[tile_id]["col"],
                "tile_id": tile_id,

                # KEY FOR THE TEMPORAL PIPELINE
                "years": [s["year"] for s in samples]
            })

        print(f"Valid sequences built: {len(sequences)}")

        if len(sequences) == 0:
            raise RuntimeError("No sequences were built.")

        return sequences

    def __len__(self) -> int:
        """Returns the total number of sequences."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns a temporal sequence at the given index.

        Args:
            idx (int): Index of the sequence to retrieve.

        Returns:
            torch.Tensor: Temporal sequence tensor of shape (T, D, H, W).
        """
        seq = self.sequences[idx]

        # (T, H, W, D) -> (T, D, H, W)
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
        drop_last=is_train
    )