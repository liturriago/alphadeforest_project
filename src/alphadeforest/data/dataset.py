import os
import json
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict


class AlphaEarthTemporalDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        years: List[int],          # ← años a usar (train o full)
        mode: str = "train",       # "train" | "full"
        transform=None,
    ):
        assert mode in ["train", "full"]

        self.dataset_dir = dataset_dir
        self.years = set(years)
        self.mode = mode
        self.transform = transform

        self.tile_metadata: List[Dict] = []
        self.sequences = self._load_sequences()

    # ==========================================================
    # 1. Cargar archivos planos desde Kaggle
    # ==========================================================
    def _load_sequences(self):

        tiles = defaultdict(list)
        coords_map = {}

        files = os.listdir(self.dataset_dir)

        json_files = [f for f in files if f.endswith(".json")]

        if len(json_files) == 0:
            raise RuntimeError("No se encontraron archivos JSON.")

        print(f"Archivos JSON detectados: {len(json_files)}")

        # ------------------------------------------------------
        # Leer metadata + embeddings
        # ------------------------------------------------------
        for json_name in json_files:

            json_path = os.path.join(self.dataset_dir, json_name)

            with open(json_path, "r") as f:
                meta = json.load(f)

            npy_name = json_name.replace(".json", ".npy")
            npy_path = os.path.join(self.dataset_dir, npy_name)

            if not os.path.exists(npy_path):
                continue

            emb = np.load(npy_path)

            # Validación defensiva crítica
            if emb.ndim != 3:
                print(f"Embedding inválido: {npy_name} → shape {emb.shape}")
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

        # ------------------------------------------------------
        # Construcción de secuencias temporales
        # ------------------------------------------------------
        sequences = []

        for tile_id in sorted(tiles.keys()):

            samples = sorted(tiles[tile_id], key=lambda x: x["year"])

            if len(samples) < 2:
                continue

            seq = np.stack([s["emb"] for s in samples], axis=0)

            # Limpieza numérica (MUY importante)
            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

            sequences.append(seq)
            self.tile_metadata.append(coords_map[tile_id])

        print(f"Secuencias válidas construidas: {len(sequences)}")

        if len(sequences) == 0:
            raise RuntimeError("No se construyeron secuencias.")

        return sequences

    # ==========================================================
    def __len__(self):
        return len(self.sequences)

    # ==========================================================
    # 2. Conversión PyTorch correcta
    # ==========================================================
    def __getitem__(self, idx):

        seq = self.sequences[idx]

        # (T, H, W, D) → (T, D, H, W)
        seq = torch.from_numpy(seq).float().permute(0, 3, 1, 2)

        if self.transform:
            seq = self.transform(seq)

        return seq


def get_dataloader(
    dataset, 
    batch_size: int, 
    num_workers: int, 
    partition: str = "train",
    pin_memory: bool = True
):

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