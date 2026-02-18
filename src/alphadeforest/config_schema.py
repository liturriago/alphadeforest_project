from pydantic import BaseModel, Field
from typing import List, Optional

class ModelConfig(BaseModel):
    embedding_dim: int = Field(64, description="Dimension of the feature vector extracted by the Encoder")
    latent_dim: int = Field(64, description="Dimension of the CAE bottleneck")
    cae_h: int = Field(16, description="Spatial height of the latent map (H)")
    cae_w: int = Field(16, description="Spatial width of the latent map (W)")
    hidden_dim_mem: int = Field(128, description="Size of the MemoryNetwork hidden layer (GRU/LSTM)")

class DataConfig(BaseModel):
    dataset_dir: str = Field(..., description="Glob path to .tar files (e.g.: data/shards-*.tar)")
    batch_size: int = Field(4, ge=1, description="Batch size (be careful with VRAM on long sequences)")
    num_workers: int = Field(4, ge=0, description="Threads for data loading (WebDataset benefits from this)")
    shuffle: bool = Field(True, description="If True, shuffles shards and samples during training")
    train_years: List[int] = Field(..., description="List of years considered 'normal' for training")
    test_years: List[int] = Field(..., description="Year to evaluate")
    mode: str = Field("train", pattern="^(train|full)$", description="Loading mode: 'train' (filters years) or 'full' (everything)")

class TrainConfig(BaseModel):
    seed: int = Field(42, description="Global seed for reproducibility (Numpy, Torch, Python)")
    epochs: int = Field(50, ge=1)
    lr: float = Field(1e-3, description="Initial learning rate")
    lambda_rec: float = Field(1.0, description="Weight of reconstruction error (spatial)")
    lambda_pred: float = Field(0.8, description="Weight of prediction error (temporal)")
    device: str = Field("cuda", description="Training device (cuda/cpu)")
    gamma: float = Field(0.9, description="Learning rate scheduler gamma")

class ExperimentConfig(BaseModel):
    name: str = Field(..., description="Experiment name")
    version: str = Field(..., description="Experiment version")
    output_dir: str = Field(..., description="Output directory for results")
    

class MainConfig(BaseModel):
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    experiment: ExperimentConfig