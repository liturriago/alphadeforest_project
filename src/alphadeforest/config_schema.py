from pydantic import BaseModel, Field
from typing import List, Optional

class ModelConfig(BaseModel):
    embedding_dim: int = Field(64, description="Dimensión del vector de características extraído por el Encoder")
    latent_dim: int = Field(64, description="Dimensión del cuello de botella (bottleneck) del CAE")
    cae_h: int = Field(16, description="Altura espacial del mapa latente (H)")
    cae_w: int = Field(16, description="Anchura espacial del mapa latente (W)")
    hidden_dim_mem: int = Field(128, description="Tamaño de la capa oculta de la MemoryNetwork (GRU/LSTM)")

class DataConfig(BaseModel):
    shards_path: str = Field(..., description="Ruta glob a los archivos .tar (ej: data/shards-*.tar)")
    batch_size: int = Field(4, ge=1, description="Tamaño del batch (cuidado con la VRAM en secuencias largas)")
    num_workers: int = Field(4, ge=0, description="Hilos para carga de datos (WebDataset se beneficia de esto)")
    shuffle: bool = Field(True, description="Si True, mezcla los shards y las muestras durante el entrenamiento")
    train_years: List[int] = Field(..., description="Lista de años considerados 'normales' para entrenar")
    mode: str = Field("train", pattern="^(train|full)$", description="Modo de carga: 'train' (filtra años) o 'full' (todo)")

class TrainConfig(BaseModel):
    seed: int = Field(42, description="Semilla global para reproducibilidad (Numpy, Torch, Python)")
    epochs: int = Field(50, ge=1)
    lr: float = Field(1e-3, description="Learning rate inicial")
    lambda_rec: float = Field(1.0, description="Peso del error de reconstrucción (espacial)")
    lambda_pred: float = Field(0.8, description="Peso del error de predicción (temporal)")
    device: str = Field("cuda", description="Dispositivo de entrenamiento (cuda/cpu)")
    
    # Directorios de salida
    checkpoint_dir: str = Field("checkpoints", description="Carpeta para guardar pesos del modelo")
    results_dir: str = Field("results", description="Carpeta para guardar mapas de anomalías e imágenes")

class MainConfig(BaseModel):
    model: ModelConfig
    data: DataConfig
    train: TrainConfig