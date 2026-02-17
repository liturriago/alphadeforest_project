"""
Reproducibility module for AlphaDeforest.
This module provides functions to set seeds for various libraries to ensure consistent results.
"""

import torch
import numpy as np
import random
import os


def set_seed(seed: int = 42) -> None:
    """
    Sets the seed for various random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    # 1. Basic seeds of Python and Numpy
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    # 2. PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    # 3. CuDNN configuration for determinism
    # Note: This may slightly reduce performance but ensures consistency
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Seeds fixed at: {seed}")