import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
from torch.amp import autocast, GradScaler
from alphadeforest.config_schema import TrainConfig
import copy 


class AlphaDeforestTrainer:

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        config: TrainConfig,
    ):
        self.device = torch.device(config.device)
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.config = config

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.best_val_loss = float("inf")

        self.history = {
            "train_loss": [], "train_rec": [], "train_pred": [],
            "val_loss": [], "val_rec": [], "val_pred": []
        }

        self.use_amp = config.use_amp and (self.device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)

        self.best_acc = 0.0
        self.best_model_wts = copy.deepcopy(model.state_dict())

    # ============================================================
    # TRAIN
    # ============================================================

    def train_epoch(self, dataloader):

        self.model.train()

        summary = {"loss": 0.0, "rec": 0.0, "pred": 0.0}

        if len(dataloader) == 0:
            print("⚠ Train loader vacío")
            return summary

        pbar = tqdm(dataloader, desc="Training")

        for batch in pbar:

            x_seq = batch.to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.device.type == "cuda"):

                outputs = self.model(x_seq)
                loss, l_rec, l_pred = self.criterion(outputs, x_seq)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            summary["loss"] += loss.item()
            summary["rec"] += l_rec.item()
            summary["pred"] += l_pred.item()

            pbar.set_postfix({
                "Total": f"{loss.item():.4f}",
                "Rec": f"{l_rec.item():.4f}",
                "Pred": f"{l_pred.item():.4f}"
            })

        n = len(dataloader)

        return {k: v / n for k, v in summary.items()}

    # ============================================================
    # EVAL
    # ============================================================

    @torch.no_grad()
    def evaluate(self, dataloader):

        self.model.eval()

        summary = {"loss": 0.0, "rec": 0.0, "pred": 0.0}

        if dataloader is None or len(dataloader) == 0:
            print("⚠ Val loader vacío")
            return summary

        for batch in dataloader:

            x_seq = batch.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):

                outputs = self.model(x_seq)
                loss, l_rec, l_pred = self.criterion(outputs, x_seq)

            summary["loss"] += loss.item()
            summary["rec"] += l_rec.item()
            summary["pred"] += l_pred.item()

        n = len(dataloader)

        return {k: v / n for k, v in summary.items()}

    # ============================================================
    # FIT
    # ============================================================

    def fit(self, train_loader, val_loader=None):

        print(f"Iniciando entrenamiento en {self.device}")

        for epoch in range(self.config.train.epochs):

            train_metrics = self.train_epoch(train_loader)

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_rec"].append(train_metrics["rec"])
            self.history["train_pred"].append(train_metrics["pred"])

            log_msg = (
                f"Epoch [{epoch+1}/{self.config.train.epochs}] "
                f"| Loss: {train_metrics['loss']:.4f} "
                f"| Rec: {train_metrics['rec']:.4f} "
                f"| Pred: {train_metrics['pred']:.4f}"
            )

            if val_loader:

                val_metrics = self.evaluate(val_loader)

                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_rec"].append(val_metrics["rec"])
                self.history["val_pred"].append(val_metrics["pred"])

                log_msg += (
                    f" | Val Loss: {val_metrics['loss']:.4f}"
                    f" | Val Rec: {val_metrics['rec']:.4f}"
                )

                # Mejor modelo
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.save_checkpoint(epoch, is_best=True)

            print(log_msg)

            # Backup periódico
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)

    # ============================================================
    # CHECKPOINT
    # ============================================================

    def save_checkpoint(self, epoch: int, is_best: bool = False):

        filename = "best_model.pt" if is_best else f"alphadeforest_epoch_{epoch+1}.pt"
        path = self.checkpoint_dir / filename

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.dict(),
            "best_val_loss": self.best_val_loss
        }, path)

        if is_best:
            print(f"Nuevo mejor modelo guardado → {path}")
        else:
            print(f"Checkpoint guardado → {path}")
