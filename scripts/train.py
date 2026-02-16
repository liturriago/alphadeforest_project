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

def run_visualization(model, dataset, config, output_dir="results"):
    print("\n🗺️  Iniciando generación de Mapas de Anomalías (Modo: FULL)...")
    model.eval()
    
    results = defaultdict(dict)
    
    # Aquí es importante: Como estamos en modo "full", el dataset puede tener
    # años que NO estaban en el entrenamiento. 
    # El dataset en modo 'full' no filtra, así que confiamos en la metadata del shard.
    
    for i in tqdm(range(len(dataset)), desc="Detectando anomalías"):
        seq_tensor = dataset[i]
        meta = dataset.tile_metadata[i] # Asegúrate de que tu dataset.py llene esto en modo full también
        
        row, col = meta['row'], meta['col']
        
        # Calculamos scores
        scores = get_anomaly_scores(
            model, 
            seq_tensor, 
            lambda_rec=config.train.lambda_rec, 
            lambda_pred=config.train.lambda_pred
        )
        
        # Mapeo de años:
        # Necesitamos saber en qué año empieza esta secuencia específica.
        # Tu dataset agrupa por tile, y ordena por año.
        # Si la secuencia en el shard es [2016, 2017, 2018, 2019]
        # score[0] -> transición 2016-2017
        # score[1] -> transición 2017-2018
        # score[2] -> transición 2018-2019
        
        # RECUPERAMOS EL AÑO INICIAL DE LA SECUENCIA DESDE EL TENSOR O METADATA
        # (Esto asume que tu dataset guarda el año inicial en metadata, 
        # si no, hay que deducirlo de config.data.train_years o del shard).
        
        # OPCIÓN ROBUSTA: Asumir continuidad desde el primer año del dataset
        start_year = min(config.data.train_years) # Ojo: esto asume que los shards empiezan aquí.
        
        for t, score in enumerate(scores):
            # El score 't' corresponde al año: start_year + (t + 1)
            target_year = start_year + t + 1
            results[target_year][(row, col)] = score

    # Guardar
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Guardando mapas en: {output_path.resolve()}")
    for year, scores_dict in results.items():
        save_anomaly_map(scores_dict, year, output_dir=str(output_path))


def main(config_path: str, run_vis: bool = True):
    # 1. Configuración
    print(f"⚙️  Cargando configuración desde {config_path}")
    with open(config_path, "r") as f:
        config = MainConfig(**yaml.safe_load(f))

    # --- APLICAR REPRODUCIBILIDAD AQUÍ ---
    print(f"🔒 Fijando semillas aleatorias: {config.train.seed}")
    set_seed(config.train.seed)
    
    # ---------------------------------------------------------
    # FASE 1: ENTRENAMIENTO (Solo años "sanos" / train_years)
    # ---------------------------------------------------------
    print("📂 [TRAIN] Cargando Dataset de Entrenamiento...")
    train_dataset = AlphaEarthTemporalDataset(
        shards_path=config.data.shards_path,
        train_years=config.data.train_years,
        mode="train"  # <--- IMPORTANTE: Solo carga lo que definiste para entrenar
    )
    
    train_loader = get_dataloader(
        train_dataset, 
        batch_size=config.data.batch_size, 
        num_workers=config.data.num_workers,
        partition="train"
    )

    print("🧠 Inicializando Modelo...")
    model = AlphaDeforest(
        embedding_dim=config.model.embedding_dim,
        latent_dim=config.model.latent_dim,
        cae_h=config.model.cae_h,
        cae_w=config.model.cae_w,
        hidden_dim_mem=config.model.hidden_dim_mem
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    trainer = AlphaDeforestTrainer(model, optimizer, config, device=config.train.device)
    
    print("🚀 Comenzando entrenamiento...")
    trainer.fit(train_loader)

    # ---------------------------------------------------------
    # FASE 2: INFERENCIA / VISUALIZACIÓN (Todo el historial)
    # ---------------------------------------------------------
    if run_vis:
        print("\n" + "="*40)
        print("🔍 INICIANDO FASE DE DETECCIÓN (INFERENCIA)")
        print("="*40)

        # Cargar mejor modelo
        try:
            checkpoint_path = Path("checkpoints/best_model.pt")
            if checkpoint_path.exists():
                print(f"🔄 Cargando pesos del mejor modelo: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=config.train.device)
                model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"⚠️ Error cargando checkpoint: {e}. Usando modelo actual.")

        # --- AQUÍ ESTÁ LA CLAVE DEL OTRO LLM ---
        print("📂 [TEST] Cargando Dataset COMPLETO (Mode='full')...")
        # Instanciamos un NUEVO dataset apuntando a los mismos .tar
        # pero con mode="full" para que cargue TODA la historia disponible
        viz_dataset = AlphaEarthTemporalDataset(
            shards_path=config.data.shards_path,
            train_years=config.data.train_years, # Se usa solo como referencia base
            mode="full"  # <--- IMPORTANTE: Carga todo sin filtrar por train_years
        )

        run_visualization(model, viz_dataset, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--no-vis", action="store_true")
    args = parser.parse_args()
    
    main(args.config, run_vis=not args.no_vis)