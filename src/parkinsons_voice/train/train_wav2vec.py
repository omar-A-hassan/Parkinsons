"""Training script for Wav2Vec2 classifier."""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

from ..data.dataset import ParkinsonsVoiceDataset, create_train_val_split, create_dataloaders
from ..models.wav2vec2_classifier import Wav2Vec2Classifier
from ..utils.config import config_loader, TrainingConfig, ModelConfig, get_device_from_config


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Wav2Vec2Trainer:
    """Trainer class for Wav2Vec2 classifier."""
    
    def __init__(
        self,
        model: Wav2Vec2Classifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: str = "auto",
        use_wandb: bool = False,
        experiment_name: str = "wav2vec2_parkinsons",
    ):
        """
        Initialize trainer.
        
        Args:
            model: Wav2Vec2Classifier model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
            use_wandb: Whether to use Weights & Biases for logging
            experiment_name: Name for the experiment
        """
        # Set device
        if device == "auto":
            device = get_device_from_config(config_loader)
        self.device = torch.device(device)
        
        # Store config
        self.config = config
        
        # Move model to device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Set up optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Set up learning rate scheduler
        total_steps = len(train_loader) * config.epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Tracking
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.experiment_name = experiment_name
        self.best_val_acc = 0.0
        self.training_history = {"train_loss": [], "val_loss": [], "val_acc": []}
        
        if self.use_wandb:
            wandb_project = config_loader.get("training_config", "logging.wandb_project", "parkinsons-voice")
            wandb.init(project=wandb_project, name=experiment_name, config=config.__dict__)
            wandb.watch(self.model)
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            input_values = batch["input_values"].to(self.device)
            labels = batch["label"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(input_values, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip_max_norm)
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            log_every = config_loader.get("training_config", "logging.log_every", 10)
            if batch_idx % log_every == 0:
                logger.info(
                    f"Batch {batch_idx}/{len(self.train_loader)}, "
                    f"Loss: {loss.item():.4f}, "
                    f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
                )
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                input_values = batch["input_values"].to(self.device)
                labels = batch["label"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Forward pass
                logits = self.model(input_values, attention_mask)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                # Predictions
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        val_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted"
        )
        
        metrics = {
            "val_loss": val_loss,
            "val_accuracy": accuracy,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
        }
        
        return metrics
    
    def train(
        self,
        num_epochs: int,
        save_dir: str = "models",
        save_best: bool = True,
        validate_every: int = 1,
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            num_epochs: Number of training epochs
            save_dir: Directory to save model checkpoints
            save_best: Whether to save the best model
            validate_every: Validate every N epochs
            
        Returns:
            Training history dictionary
        """
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch()
            self.training_history["train_loss"].append(train_loss)
            
            # Validation
            if (epoch + 1) % validate_every == 0:
                metrics = self.validate()
                val_loss = metrics["val_loss"]
                val_acc = metrics["val_accuracy"]
                
                self.training_history["val_loss"].append(val_loss)
                self.training_history["val_acc"].append(val_acc)
                
                logger.info(
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_acc:.4f}"
                )
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        **metrics,
                    })
                
                # Save best model
                if save_best and val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    best_model_path = Path(save_dir) / "best_model.pt"
                    self.save_checkpoint(best_model_path, epoch, metrics)
                    logger.info(f"New best model saved with validation accuracy: {val_acc:.4f}")
            
            # Save latest checkpoint
            latest_model_path = Path(save_dir) / "latest_model.pt"
            self.save_checkpoint(latest_model_path, epoch, {"train_loss": train_loss})
        
        logger.info("Training completed!")
        
        if self.use_wandb:
            wandb.finish()
        
        return self.training_history
    
    def save_checkpoint(
        self, 
        path: Path, 
        epoch: int, 
        metrics: Dict[str, float]
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "training_history": self.training_history,
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.training_history = checkpoint["training_history"]
        return checkpoint


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Wav2Vec2 classifier for Parkinson's detection")
    parser.add_argument("--config_dir", type=str, default="config",
                       help="Directory containing configuration files")
    parser.add_argument("--data_csv", type=str, default=None,
                       help="Path to dataset CSV file (overrides config)")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging (overrides config)")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Experiment name for wandb")
    
    args = parser.parse_args()
    
    # Load configurations
    global config_loader
    config_loader = config_loader.__class__(args.config_dir)
    
    training_config = TrainingConfig.from_config(config_loader)
    model_config = ModelConfig.from_config(config_loader)
    
    # Override with command line arguments
    data_csv = args.data_csv or config_loader.get("data_config", "paths.dataset_csv", "data/dataset_metadata.csv")
    use_wandb = args.use_wandb or config_loader.get("training_config", "logging.use_wandb", False)
    experiment_name = args.experiment_name or config_loader.get("training_config", "logging.wandb_name", "wav2vec2_parkinsons")
    save_dir = config_loader.get("training_config", "saving.save_dir", "models")
    
    # Set up dataset
    logger.info("Loading dataset...")
    dataset = ParkinsonsVoiceDataset(
        csv_path=data_csv,
        processor_name=model_config.model_name,
        target_sample_rate=model_config.target_sample_rate,
        max_duration=model_config.max_duration,
        normalize_audio=model_config.normalize_audio,
    )
    
    logger.info(f"Dataset loaded with {len(dataset)} samples")
    logger.info(f"Class distribution: {dataset.get_class_distribution()}")
    
    # Create train/validation split
    train_dataset, val_dataset = create_train_val_split(
        dataset, val_split=training_config.val_split, stratify=training_config.stratify
    )
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, 
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
    )
    
    # Initialize model
    model = Wav2Vec2Classifier(
        model_name=model_config.model_name,
        num_classes=model_config.num_classes,
        freeze_feature_extractor=model_config.freeze_feature_extractor,
        freeze_transformer=model_config.freeze_transformer,
        dropout_rate=model_config.dropout_rate,
        hidden_dim=model_config.hidden_dim,
    )
    
    # Initialize trainer
    trainer = Wav2Vec2Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        use_wandb=use_wandb,
        experiment_name=experiment_name,
    )
    
    # Train model
    history = trainer.train(
        num_epochs=training_config.epochs,
        save_dir=save_dir,
        validate_every=training_config.validate_every,
    )
    
    # Save training history
    history_path = Path(save_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Training history saved to {history_path}")
    logger.info(f"Best validation accuracy: {trainer.best_val_acc:.4f}")


if __name__ == "__main__":
    main()