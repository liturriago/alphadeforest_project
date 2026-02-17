"""
Training script for the AlphaDeforest model.
This script loads the configuration, initializes the datasets, model, and trainer,
and starts the training process.
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from typing import Optional

# Project imports
from alphadeforest.config_schema import MainConfig
from alphadeforest.data.dataset import get_dataloader, AlphaEarthTemporalDataset
from alphadeforest.models.alpha_deforest import AlphaDeforest
from alphadeforest.engine.trainer import AlphaDeforestTrainer
from alphadeforest.losses.rpc import RPCLoss
from alphadeforest.utils.visualizer import generate_anomaly_maps
from alphadeforest.utils.reproducibility import set_seed


def main(config_path: str, run_vis: bool = True) -> None:
    """
    Main training function.

    Args:
        config_path (str): Path to the YAML configuration file.
        run_vis (bool): Whether to run visualization after training. Defaults to True.
    """

    print(f"Loading configuration from {config_path}")
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
        config = MainConfig(**config_data)

    print(f" Setting random seeds: {config.train.seed}")
    set_seed(config.train.seed)
    
    print("[TRAIN] Loading Training Dataset...")
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

    print("[TEST] Loading FULL Dataset (Mode='test')...")
    test_dataset = AlphaEarthTemporalDataset(
        dataset_dir=config.data.dataset_dir,
        years=config.data.test_year,
        mode="test" 
    )

    test_loader = get_dataloader(
        test_dataset, 
        batch_size=config.data.batch_size, 
        num_workers=config.data.num_workers,
        partition="test"
    )

    print("Initializing Model...")
    model = AlphaDeforest(
        embedding_dim=config.model.embedding_dim,
        latent_dim=config.model.latent_dim,
        cae_h=config.model.cae_h,
        cae_w=config.model.cae_w,
        hidden_dim_mem=config.model.hidden_dim_mem
    )

    criterion = RPCLoss(
        lambda_rec=config.train.lambda_rec,
        lambda_pred=config.train.lambda_pred
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.train.gamma)
    
    # Validation loader is set to None as per original script
    trainer = AlphaDeforestTrainer(model, train_loader, None, criterion, optimizer, config.train)
    
    print("Beginning training...")
    trained_model = trainer.fit(scheduler=scheduler)

    output_dir = config.experiment.output_dir

    if run_vis:
        print("\n" + "="*60)
        print("GENERATING ANOMALY HEATMAPS")
        print("="*60)

        generate_anomaly_maps(
            model=model,
            dataloader=test_loader,
            device=config.train.device,
            output_dir=output_dir,
            use_prediction=True,
            lambda_rec=1.0,
            lambda_pred=0.5,
        )

    save_path = Path(output_dir) / "model_final.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trained_model.state_dict(), save_path)
    print(f"\n Experiment completed. Results saved in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaDeforest Training Script")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to config file")
    parser.add_argument("--vis", action="store_true", help="Run visualization after training")
    args = parser.parse_args()
    
    main(args.config, run_vis=args.vis)