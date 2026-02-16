import io
import json
import numpy as np
import torch
import webdataset as wds
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from typing import List

class AlphaEarthTemporalDataset(Dataset):
    def __init__(
        self,
        shards_path: str,
        train_years: List[int],
        mode: str = "train",   # "train" | "full"
        transform=None,
    ):
        assert mode in ["train", "full"]
        self.shards_path = shards_path
        self.train_years = set(train_years)
        self.mode = mode
        self.transform = transform

        # --- NUEVO: Para mapeo geográfico ---
        self.tile_metadata = [] 
        self.sequences = self._load_sequences()

    def _load_sequences(self):
        tiles = defaultdict(list)
        # Diccionario auxiliar para guardar metadata por tile_id
        coords_map = {} 

        dataset = wds.WebDataset(self.shards_path, shardshuffle=False)

        # 1. Leer shards y agrupar por tile
        for sample in dataset:
            meta = json.loads(
                sample["meta.json"].decode("utf-8")
                if isinstance(sample["meta.json"], bytes)
                else sample["meta.json"]
            )
            emb = np.load(io.BytesIO(sample["emb.npy"]))

            tile_id = meta["tile_id"]
            tiles[tile_id].append({
                "year": meta["year"],
                "emb": emb
            })
            
            # Guardamos las coordenadas una sola vez por tile_id
            if tile_id not in coords_map:
                coords_map[tile_id] = {
                    "row": meta.get("row", 0),
                    "col": meta.get("col", 0),
                    "tile_id": tile_id
                }

        # 2. Construir secuencias temporales
        sequences = []
        for tile_id in sorted(tiles.keys()):
            samples = sorted(tiles[tile_id], key=lambda x: x["year"])

            if self.mode == "train":
                samples = [s for s in samples if s["year"] in self.train_years]

            if len(samples) < 2:
                continue

            # --- NUEVO: Guardar metadata en el mismo orden que las secuencias ---
            self.tile_metadata.append(coords_map[tile_id])
            
            seq = np.stack([s["emb"] for s in samples], axis=0)  # (T, H, W, D)
            sequences.append(seq)

        print(f"✅ Dataset cargado: {len(sequences)} secuencias encontradas.")
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # (T, H, W, D) -> (T, D, H, W)
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