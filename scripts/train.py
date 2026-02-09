import argparse
import yaml
import torch
from alphadeforest.config_schema import MainConfig
from alphadeforest.data.dataset import get_dataloader, AlphaEarthTemporalDataset
from alphadeforest.models.alpha_deforest import AlphaDeforest
from alphadeforest.engine.trainer import AlphaDeforestTrainer

def main(config_path: str):
    # 1. Cargar y validar configuración
    with open(config_path, "r") as f:
        config = MainConfig(**yaml.safe_load(f))
    
    # 2. Preparar Datos
    # Nota: Aquí asumo que ya descargaste los datos con kagglehub
    dataset = AlphaEarthTemporalDataset(config.data.shards_path,
                                        train_years=config.data.train_years,
                                        mode=config.data.mode                       
    )
    
    train_loader = get_dataloader(
        dataset, 
        batch_size=config.data.batch_size, 
        num_workers=config.data.num_workers,
        partition="train"
    )

    # 3. Inicializar Modelo
    model = AlphaDeforest(
        embedding_dim=config.model.embedding_dim,
        latent_dim=config.model.latent_dim,
        cae_h=config.model.cae_h,
        cae_w=config.model.cae_w,
        hidden_dim_mem=config.model.hidden_dim_mem
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)

    # 4. Entrenar
    trainer = AlphaDeforestTrainer(model, optimizer, config, device=config.train.device)
    trainer.fit(train_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()
    main(args.config)