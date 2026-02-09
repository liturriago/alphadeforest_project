import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any

class AlphaDeforestTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Any,  # MainConfig de Pydantic
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # --- NUEVO: Tracking del mejor modelo ---
        self.best_val_loss = float('inf')
        
        self.history = {
            "train_loss": [], "train_rec": [], "train_pred": [],
            "val_loss": [], "val_rec": []
        }

    def _compute_loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor):
        """
        Calcula la pérdida combinada: L_total = λ_rec * L_rec + λ_pred * L_pred
        """
        # 1. Error de Reconstrucción (Espacial)
        x_rec = outputs["reconstructions"]
        loss_rec = F.mse_loss(x_rec, targets)

        # 2. Error de Predicción (Temporal)
        # z_f: (B, T, Z) -> tomamos desde t=1 para comparar con z_pred
        z_f_target = outputs["z_f"][:, 1:]
        z_pred = outputs["z_pred"]
        loss_pred = F.mse_loss(z_pred, z_f_target)

        total_loss = (self.config.train.lambda_rec * loss_rec + 
                      self.config.train.lambda_pred * loss_pred)
        
        return total_loss, loss_rec, loss_pred

    def train_epoch(self, dataloader):
        self.model.train()
        summary = {"loss": 0, "rec": 0, "pred": 0}
        
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            x_seq = batch.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(x_seq)
            
            loss, l_rec, l_pred = self._compute_loss(outputs, x_seq)
            
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
            loss, l_rec, l_pred = self._compute_loss(outputs, x_seq)
            
            summary["loss"] += loss.item()
            summary["rec"] += l_rec.item()
            summary["pred"] += l_pred.item()
            
        n = len(dataloader)
        return {k: v / n for k, v in summary.items()}

    def fit(self, train_loader, val_loader=None):
        print(f"🚀 Iniciando entrenamiento en {self.device} por {self.config.train.epochs} épocas")
        
        for epoch in range(self.config.train.epochs):
            train_metrics = self.train_epoch(train_loader)
            
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_rec"].append(train_metrics["rec"])
            self.history["train_pred"].append(train_metrics["pred"])
            
            log_msg = f"Epoch [{epoch+1}/{self.config.train.epochs}] | Loss: {train_metrics['loss']:.4f}"
            
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_rec"].append(val_metrics["rec"])
                log_msg += f" | Val Loss: {val_metrics['loss']:.4f}"
                
                # --- NUEVO: Lógica de guardado del mejor modelo ---
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.save_checkpoint(epoch, is_best=True)
            
            print(log_msg)
            
            # Guardar backup regular
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        filename = "best_model.pt" if is_best else f"alphadeforest_epoch_{epoch+1}.pt"
        path = self.checkpoint_dir / filename
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.dict(),
            'best_loss': self.best_val_loss if is_best else None
        }, path)
        
        if is_best:
            print(f"⭐ ¡Nuevo mejor modelo guardado!: {path}")
        else:
            print(f"💾 Checkpoint regular guardado: {path}")