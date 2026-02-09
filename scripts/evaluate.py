import argparse
import yaml
import torch
import os
from tqdm import tqdm
from collections import defaultdict
from alphadeforest.config_schema import MainConfig
from alphadeforest.models.alpha_deforest import AlphaDeforest
from alphadeforest.engine.inference import get_anomaly_scores
from alphadeforest.data.dataset import AlphaEarthTemporalDataset
from alphadeforest.utils.visualizer import save_anomaly_map

def main(config_path: str, checkpoint_path: str):
    # 1. Load and validate configuration
    with open(config_path, "r") as f:
        config = MainConfig(**yaml.safe_load(f))
    
    # 2. Setup Device and Model
    device = torch.device(config.train.device if torch.cuda.is_available() else "cpu")
    
    model = AlphaDeforest(
        embedding_dim=config.model.embedding_dim,
        latent_dim=config.model.latent_dim,
        cae_h=config.model.cae_h,
        cae_w=config.model.cae_w,
        hidden_dim_mem=config.model.hidden_dim_mem
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']+1}")

    # 3. Load Dataset in Test Mode
    # Note: We need a dataset version that preserves tile metadata
    dataset = AlphaEarthTemporalDataset(
        config.data.shards_path,
        train_years=config.data.train_years,
        mode="test"
    )

    # 4. Inference and Score Collection
    # {year: {(row, col): score}}
    results_by_year = defaultdict(dict)

    print(f"🔍 Analyzing {len(dataset)} tiles for anomaly mapping...")
    for i in tqdm(range(len(dataset))):
        tile_seq = dataset[i]
        # Coordinates must be exposed by your dataset class
        meta = dataset.tile_metadata[i] 
        
        scores = get_anomaly_scores(
            model, 
            tile_seq, 
            lambda_rec=config.train.lambda_rec,
            lambda_pred=config.train.lambda_pred
        )
        
        # scores[0] corresponds to index 1 of train_years
        for t_idx, score in enumerate(scores):
            target_year = config.data.train_years[t_idx + 1]
            coords = (meta['row'], meta['col'])
            results_by_year[target_year][coords] = score

    # 5. Generate and Save Pseudo-Maps
    for year, scores_dict in results_by_year.items():
        save_anomaly_map(scores_dict, year, output_dir="results/maps")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference and Mapping Script")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt model file")
    
    args = parser.parse_args()
    main(args.config, args.checkpoint)