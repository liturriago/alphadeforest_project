"""
Configuration schema for the AlphaDeforest project.
This module defines the data structures for model, training, and experiment configuration
using Pydantic models for validation and type safety.
"""

from pydantic import BaseModel, Field
from typing import List, Optional

class ModelConfig(BaseModel):
    """
    Configuration for the AlphaDeforest model architecture.
    """
    embedding_dim: int = Field(64, description="Dimension of the feature vector extracted by the Encoder")
    latent_dim: int = Field(64, description="Dimension of the CAE bottleneck")
    cae_h: int = Field(16, description="Spatial height of the latent map (H)")
    cae_w: int = Field(16, description="Spatial width of the latent map (W)")
    hidden_dim_mem: int = Field(128, description="Hidden layer size of the MemoryNetwork (GRU/LSTM)")

class DataConfig(BaseModel):
    """
    Configuration for data loading and dataset parameters.
    """
    dataset_dir: str = Field(..., description="Glob path to .tar files (e.g., data/shards-*.tar)")
    batch_size: int = Field(4, ge=1, description="Batch size (beware of VRAM with long sequences)")
    num_workers: int = Field(4, ge=0, description="Threads for data loading (WebDataset benefits from this)")
    shuffle: bool = Field(True, description="Whether to shuffle shards and samples during training")
    train_years: List[int] = Field(..., description="List of years considered 'normal' for training")
    test_year: List[int] = Field(..., description="Year(s) to evaluate")

class TrainConfig(BaseModel):
    """
    Configuration for the training process.
    """
    seed: int = Field(42, description="Global seed for reproducibility (Numpy, Torch, Python)")
    epochs: int = Field(50, ge=1, description="Number of training epochs")
    lr: float = Field(1e-3, description="Initial learning rate")
    gamma: float = Field(0.94, description="Learning rate decay factor")
    lambda_rec: float = Field(1.0, description="Weight of the reconstruction error (spatial)")
    lambda_pred: float = Field(0.8, description="Weight of the prediction error (temporal)")
    device: str = Field("cuda", description="Training device (cuda/cpu)")
    use_amp: bool = Field(True, description="Whether to use mixed precision (Automatic Mixed Precision)") 
    
class ExperimentConfig(BaseModel):
    """
    Configuration for experiment metadata and output.
    """
    name: str = Field(..., description="Name of the experiment")
    version: int = Field(..., description="Version number of the experiment")
    output_dir: str = Field(..., description="Directory to save experiment results")
    
class MainConfig(BaseModel):
    """
    Root configuration object containing all sub-configs.
    """
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    experiment: ExperimentConfig