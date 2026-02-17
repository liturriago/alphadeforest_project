import os
import time
import torch
import copy
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
from alphadeforest.config_schema import TrainConfig

class AlphaDeforestTrainer:
    def __init__(self, model, train_loader, valid_loader, criterion, optimizer, config: TrainConfig):
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
        """Converts seconds to MM:SS format."""
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}" 

    def train_epoch(self, dataloader):
        self.model.train()
        summary = {"loss": 0, "rec": 0, "pred": 0}
        
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            x_seq = batch.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(x_seq)
            
            loss, l_rec, l_pred = self.criterion(outputs, x_seq)
            
            loss.backward()
            self.optimizer.step()
            
            summary["loss"] += loss.item()
            summary["rec"] += l_rec.item()
            summary["pred"] += l_pred.item()
            
            pbar.set_postfix({"Total": f"{loss.item():.4f}", "Rec": f"{l_rec.item():.4f}"})
            
        n = len(dataloader)
        return {k: v / n for k, v in summary.items()}

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        summary = {"loss": 0, "rec": 0, "pred": 0}
        
        for batch in dataloader:
            x_seq = batch.to(self.device)
            outputs = self.model(x_seq)
            loss, l_rec, l_pred = self.criterion(outputs, x_seq)
            
            summary["loss"] += loss.item()
            summary["rec"] += l_rec.item()
            summary["pred"] += l_pred.item()
            
        n = len(dataloader)
        return {k: v / n for k, v in summary.items()}

    def fit(self, scheduler=None):
        total_train_start = time.time()
        print(f"Iniciando entrenamiento en {self.device} por {self.config.epochs} épocas")
        
        for epoch in range(self.config.epochs):
            train_metrics = self.train_epoch(train_loader)
            
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_rec"].append(train_metrics["rec"])
            self.history["train_pred"].append(train_metrics["pred"])
            
            print(f"Epoch [{epoch+1}/{self.config.train.epochs}] | Loss: {train_metrics['loss']:.4f}")
            
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_rec"].append(val_metrics["rec"])
                
                if val_metrics["loss"] < self.best_loss:
                    self.best_loss = val_metrics["loss"]
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())

            else:
                if train_metrics["loss"] < self.best_loss:
                    self.best_loss = train_metrics["loss"]
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())

            if scheduler: scheduler.step()
        
        total_time = time.time() - total_train_start
        print(f"\n{' TRAINING COMPLETE ':=^50}")
        print(f"Total Duration: {self._format_time(total_time)}")
        print(f"Best Target Accuracy: {self.best_loss:.4f}")
        print("="*50)

        self.model.load_state_dict(self.best_model_wts)
        return self.model