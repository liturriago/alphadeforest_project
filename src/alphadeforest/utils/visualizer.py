import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

def save_anomaly_map(scores_dict, year, output_dir="results/maps"):
    os.makedirs(output_dir, exist_ok=True)

    rows = max(k[0] for k in scores_dict.keys()) + 1
    cols = max(k[1] for k in scores_dict.keys()) + 1

    heatmap = np.full((rows, cols), np.nan)
    for (r, c), score in scores_dict.items():
        heatmap[r, c] = score

    plt.figure(figsize=(10, 8))
    cmap = mpl.colormaps["YlOrRd"].copy() 
    cmap.set_bad(color="black")

    im = plt.imshow(heatmap, cmap=cmap)
    plt.colorbar(im, label="Deforestation Anomaly Score")
    plt.title(f"Anomaly Map - {year}")
    plt.xlabel("Tile Column")
    plt.ylabel("Tile Row")

    path = os.path.join(output_dir, f"anomaly_map_{year}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Map saved: {path}")
