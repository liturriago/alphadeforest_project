from pydantic import BaseModel, Field
from typing import List

class ModelConfig(BaseModel):
    embedding_dim: int = 64
    latent_dim: int = 64
    cae_h: int = 16
    cae_w: int = 16
    hidden_dim_mem: int = 128

class DataConfig(BaseModel):
    shards_path: str
    batch_size: int = 2
    num_workers: int = 2
    shuffle: bool = True
    train_years: List[int]
    mode: str = "train"

class TrainConfig(BaseModel):
    epochs: int = 50
    lr: float = 1e-3
    lambda_rec: float = 1.0
    lambda_pred: float = 0.5
    device: str = "cuda"

class MainConfig(BaseModel):
    model: ModelConfig
    data: DataConfig
    train: TrainConfig