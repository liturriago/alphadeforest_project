"""
Trainer module for the AlphaDeforest model.
This module handles the training and evaluation loops.
"""

import os
import time
import torch
import copy
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, Optional, List
from alphadeforest.config_schema import TrainConfig
from torch.amp import GradScaler, autocast

class AlphaDeforestTrainer:
    """
    Trainer class for the AlphaDeforest model.
    Handles training epochs, evaluation, and the overall fit process.
    """
    def __init__(
        self, 
        model: torch.nn.Module, 
        train_loader: torch.utils.data.DataLoader, 
        valid_loader: Optional[torch.utils.data.DataLoader], 
        criterion: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        config: TrainConfig
    ):
        """
        Initializes the trainer.

        Args:
            model (torch.nn.Module): The model to train.
            train_loader (DataLoader): DataLoader for the training set.
            valid_loader (Optional[DataLoader]): DataLoader for the validation set.
            criterion (torch.nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer.
            config (TrainConfig): Training configuration.
        """
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(config.device)

        self.use_amp = config.use_amp and (self.device.type == 'cuda')
        self.scaler = GradScaler(enabled=self.use_amp)
        
        self.best_loss = float('inf')
        
        self.history = {
            "train_loss": [], "train_rec": [], "train_pred": [],
            "val_loss": [], "val_rec": []
        }

        self.best_model_wts = copy.deepcopy(model.state_dict())

    def _format_time(self, seconds: float) -> str:
        """
        Converts seconds to MM:SS format.

        Args:
            seconds (float): Time in seconds.

        Returns:
            str: Formatted time string.
        """
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}" 

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Runs a single training epoch.

        Args:
            dataloader (DataLoader): The training DataLoader.

        Returns:
            Dict[str, float]: Average metrics for the epoch.
        """
        self.model.train()
        summary = {"loss": 0.0, "rec": 0.0, "pred": 0.0}
        
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            x_seq = batch.to(self.device, dtype=torch.float32)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp, device_type=self.config.device):
                outputs = self.model(x_seq)
                loss, l_rec, l_pred = self.criterion(outputs, x_seq)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            summary["loss"] += loss.item()
            summary["rec"] += l_rec.item()
            summary["pred"] += l_pred.item()
            
            pbar.set_postfix({"Total": f"{loss.item():.4f}", "Rec": f"{l_rec.item():.4f}"})
            
        n = len(dataloader)
        return {k: v / n for k, v in summary.items()}

    @torch.no_grad()
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Runs evaluation on the given dataloader.

        Args:
            dataloader (DataLoader): The evaluation DataLoader.

        Returns:
            Dict[str, float]: Average metrics for the evaluation.
        """
        self.model.eval()
        summary = {"loss": 0.0, "rec": 0.0, "pred": 0.0}
        
        for batch in dataloader:
            x_seq = batch.to(self.device, dtype=torch.float32)
            with autocast(enabled=self.use_amp, device_type=self.config.device):
                outputs = self.model(x_seq)
                loss, l_rec, l_pred = self.criterion(outputs, x_seq)
            
            summary["loss"] += loss.item()
            summary["rec"] += l_rec.item()
            summary["pred"] += l_pred.item()
            
        n = len(dataloader)
        return {k: v / n for k, v in summary.items()}

    def fit(self, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> torch.nn.Module:
        """
        Runs the full training process for the number of epochs specified in config.

        Args:
            scheduler (Optional[_LRScheduler]): Learning rate scheduler.

        Returns:
            torch.nn.Module: The model with the best weights loaded.
        """
        total_train_start = time.time()
        print(f"Starting training on {self.device} for {self.config.epochs} epochs")
        
        for epoch in range(self.config.epochs):
            train_metrics = self.train_epoch(self.train_loader)
            
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_rec"].append(train_metrics["rec"])
            self.history["train_pred"].append(train_metrics["pred"])
            
            print(f"Epoch [{epoch+1}/{self.config.epochs}] | Loss: {train_metrics['loss']:.4f}")
            
            if self.valid_loader:
                val_metrics = self.evaluate(self.valid_loader)
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_rec"].append(val_metrics["rec"])
                
                if val_metrics["loss"] < self.best_loss:
                    self.best_loss = val_metrics["loss"]
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())

            else:
                if train_metrics["loss"] < self.best_loss:
                    self.best_loss = train_metrics["loss"]
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())

            if scheduler: 
                scheduler.step()
        
        total_time = time.time() - total_train_start
        print(f"\n{' TRAINING COMPLETE ':=^50}")
        print(f"Total Duration: {self._format_time(total_time)}")
        print(f"Best Target Accuracy (Loss): {self.best_loss:.4f}")
        print("="*50)

        self.model.load_state_dict(self.best_model_wts)
        return self.model