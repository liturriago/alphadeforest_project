import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Imports de tu proyecto
from alphadeforest.config_schema import MainConfig
from alphadeforest.data.dataset import get_dataloader, AlphaEarthTemporalDataset
from alphadeforest.models.alpha_deforest import AlphaDeforest
from alphadeforest.engine.trainer import AlphaDeforestTrainer
from alphadeforest.utils.anomaly import get_anomaly_scores
from alphadeforest.utils.visualizer import save_anomaly_map

def main(config_path: str, run_vis: bool = True):

    print(f"Cargando configuración desde {config_path}")
    with open(config_path, "r") as f:
        config = MainConfig(**yaml.safe_load(f))

    print(f" Fijando semillas aleatorias: {config.train.seed}")
    set_seed(config.train.seed)
    

    print("[TRAIN] Cargando Dataset de Entrenamiento...")
    train_dataset = AlphaEarthTemporalDataset(
        dataset_dir=config.data.dataset_dir,
        years=config.data.train_years,
        mode="train" 
    )
    
    train_loader = get_dataloader(
        train_dataset, 
        batch_size=config.data.batch_size, 
        num_workers=config.data.num_workers,
        partition="train"
    )

    print("[TEST] Cargando Dataset COMPLETO (Mode='full')...")
    test_dataset = AlphaEarthTemporalDataset(
        dataset_dir=config.data.dataset_dir,
        years=config.data.test_year,
        mode="full" 
    )

    test_loader = get_dataloader(
        test_dataset, 
        batch_size=config.data.batch_size, 
        num_workers=config.data.num_workers,
        partition="test"
    )

    print("Inicializando Modelo...")
    model = AlphaDeforest(
        embedding_dim=config.model.embedding_dim,
        latent_dim=config.model.latent_dim,
        cae_h=config.model.cae_h,
        cae_w=config.model.cae_w,
        hidden_dim_mem=config.model.hidden_dim_mem
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=config.train.gamma)
    trainer = AlphaDeforestTrainer(model, optimizer, config, device=config.train.device)
    
    print("Comenzando entrenamiento...")
    trainer.fit(train_loader, scheduler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--no-vis", action="store_true")
    args = parser.parse_args()
    
    main(args.config, run_vis=not args.no_vis)