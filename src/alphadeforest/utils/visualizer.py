"""
Visualization module for AlphaDeforest.
This module provides functions to generate anomaly maps from model outputs.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import os
from typing import Any, Optional, Dict, Tuple


def generate_anomaly_maps(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
    output_dir: str = "results/maps",
    use_prediction: bool = False,
    lambda_rec: float = 1.0,
    lambda_pred: float = 0.5,
) -> None:
    """
    Generates anomaly heatmaps for each year based on model scores.

    Args:
        model (torch.nn.Module): The trained AlphaDeforest model.
        dataloader (DataLoader): DataLoader for the test/evaluation set.
        device (str): Device to run inference on. Defaults to "cuda".
        output_dir (str): Directory to save the generated maps. Defaults to "results/maps".
        use_prediction (bool): Whether to include prediction error in the score. Defaults to False.
        lambda_rec (float): Weight for the reconstruction error. Defaults to 1.0.
        lambda_pred (float): Weight for the prediction error. Defaults to 0.5.
    """

    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to store scores: {year -> {(row, col): score}}
    anomaly_maps: Dict[int, Dict[Tuple[int, int], float]] = defaultdict(dict)

    with torch.no_grad():

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Visualizing")):

            x_seq = batch.to(device)   # (B, T, C, H, W)
            outputs = model(x_seq)

            recon_error = outputs["recon_error"].cpu().numpy()  # (B, T)

            if use_prediction:
                z_f = outputs["z_f"]
                z_pred = outputs["z_pred"]

            batch_size = x_seq.shape[0]

            for b in range(batch_size):
                # Retrieve metadata for the current sample in the batch
                meta = dataloader.dataset.tile_metadata[
                    batch_idx * batch_size + b
                ]

                row, col = meta["row"], meta["col"]
                years = meta["years"]

                for t, year in enumerate(years):

                    rec_score = recon_error[b, t]

                    if use_prediction and t > 0:
                        # Calculate prediction error in latent space
                        pred_score = torch.nn.functional.mse_loss(
                            z_pred[b, t - 1],
                            z_f[b, t].view(-1)
                        ).item()

                        score = (
                            lambda_rec * rec_score +
                            lambda_pred * pred_score
                        )
                    else:
                        score = rec_score

                    anomaly_maps[year][(row, col)] = score

    # Construct and save maps for each year
    for year, scores_dict in anomaly_maps.items():

        max_row = max(k[0] for k in scores_dict.keys()) + 1
        max_col = max(k[1] for k in scores_dict.keys()) + 1

        heatmap = np.full((max_row, max_col), np.nan)

        for (r, c), score in scores_dict.items():
            heatmap[r, c] = score

        plt.figure(figsize=(10, 8))

        # Color map configuration
        cmap = plt.get_cmap("YlOrRd").copy()
        cmap.set_bad(color="black")

        im = plt.imshow(heatmap, cmap=cmap)
        plt.colorbar(im, label="Anomaly Score")

        plt.title(f"Anomaly Map - Year {year}")
        plt.xlabel("Tile Column")
        plt.ylabel("Tile Row")

        save_path = os.path.join(output_dir, f"map_{year}.png")

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved -> {save_path}")
