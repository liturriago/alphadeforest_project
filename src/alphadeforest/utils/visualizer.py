import matplotlib.pyplot as plt
import numpy as np
import os

def save_anomaly_map(scores_dict, year, output_dir="results/maps"):
    """
    Assembles a heatmap from sparse tile coordinates.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate grid bounds
    rows = max(k[0] for k in scores_dict.keys()) + 1
    cols = max(k[1] for k in scores_dict.keys()) + 1
    
    # Create the grid
    heatmap = np.full((rows, cols), np.nan) # nan for empty areas
    for (r, c), score in scores_dict.items():
        heatmap[r, c] = score
        
    plt.figure(figsize=(12, 10))
    current_cmap = plt.cm.get_cmap('YlOrRd').copy()
    current_cmap.set_bad(color='black') # Background for areas without data
    
    im = plt.imshow(heatmap, cmap=current_cmap)
    plt.colorbar(im, label='Deforestation Anomaly Score')
    plt.title(f"Reconstructed Anomaly Map - Year {year}")
    plt.xlabel("Tile Column Index")
    plt.ylabel("Tile Row Index")
    
    save_path = os.path.join(output_dir, f"anomaly_map_{year}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 Pseudo-map for {year} saved to {save_path}")