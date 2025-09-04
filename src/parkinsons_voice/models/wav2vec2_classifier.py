"""Wav2Vec2-based classifier for Parkinson's disease detection."""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import joblib
from pathlib import Path


class Wav2Vec2Classifier(nn.Module):
    """Wav2Vec2-based neural classifier for binary classification."""
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        num_classes: int = 2,
        freeze_feature_extractor: bool = False,
        freeze_transformer: bool = False,
        dropout_rate: float = 0.1,
        hidden_dim: int = 256,
    ):
        """
        Initialize Wav2Vec2 classifier.
        
        Args:
            model_name: Pre-trained Wav2Vec2 model name
            num_classes: Number of output classes
            freeze_feature_extractor: Whether to freeze the feature extractor
            freeze_transformer: Whether to freeze the transformer layers
            dropout_rate: Dropout rate for classification head
            hidden_dim: Hidden dimension for classification head
        """
        super().__init__()
        
        # Load pre-trained Wav2Vec2 model
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.config = self.wav2vec2.config
        
        # Freeze components if specified
        if freeze_feature_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()
        
        if freeze_transformer:
            for param in self.wav2vec2.encoder.parameters():
                param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )
        
        self.num_classes = num_classes
        
    def forward(
        self, 
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_values: Input audio tensor [batch_size, seq_length]
            attention_mask: Attention mask tensor
            
        Returns:
            Logits tensor [batch_size, num_classes]
        """
        # Extract features using Wav2Vec2
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        
        # Global average pooling over the sequence dimension
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        if attention_mask is not None:
            # Wav2Vec2 output has different sequence length than input
            # We need to compute the actual sequence length after feature extraction
            seq_len = hidden_states.size(1)
            
            # Create a mask for the actual output sequence length
            if attention_mask.size(1) != seq_len:
                # Approximate the attention mask for the output sequence
                # Wav2Vec2 typically reduces sequence length by ~320x factor
                reduction_factor = attention_mask.size(1) / seq_len
                indices = torch.arange(seq_len, device=attention_mask.device) * reduction_factor
                indices = indices.long().clamp(0, attention_mask.size(1) - 1)
                attention_mask = attention_mask.gather(1, indices.unsqueeze(0).expand(attention_mask.size(0), -1))
            
            # Apply attention mask before pooling
            attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            hidden_states = hidden_states * attention_mask
            pooled = hidden_states.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        
        # Apply dropout and classify
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits
    
    def extract_features(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract features without classification.
        
        Args:
            input_values: Input audio tensor
            attention_mask: Attention mask tensor
            
        Returns:
            Feature tensor [batch_size, hidden_size]
        """
        with torch.no_grad():
            outputs = self.wav2vec2(
                input_values=input_values,
                attention_mask=attention_mask,
            )
            
            hidden_states = outputs.last_hidden_state
            
            if attention_mask is not None:
                # Handle sequence length mismatch between input and output
                seq_len = hidden_states.size(1)
                
                if attention_mask.size(1) != seq_len:
                    # Approximate the attention mask for the output sequence
                    reduction_factor = attention_mask.size(1) / seq_len
                    indices = torch.arange(seq_len, device=attention_mask.device) * reduction_factor
                    indices = indices.long().clamp(0, attention_mask.size(1) - 1)
                    attention_mask = attention_mask.gather(1, indices.unsqueeze(0).expand(attention_mask.size(0), -1))
                
                attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                hidden_states = hidden_states * attention_mask
                pooled = hidden_states.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = hidden_states.mean(dim=1)
            
            return pooled


class Wav2Vec2FeatureExtractor:
    """Wav2Vec2 feature extractor for use with external classifiers."""
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        device: str = "auto",
    ):
        """
        Initialize feature extractor.
        
        Args:
            model_name: Pre-trained Wav2Vec2 model name
            device: Device to run the model on
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = torch.device(device)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def extract_features(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Extract features from audio.
        
        Args:
            input_values: Input audio tensor
            attention_mask: Attention mask tensor
            
        Returns:
            Feature array [batch_size, hidden_size]
        """
        with torch.no_grad():
            if input_values.device != self.device:
                input_values = input_values.to(self.device)
            
            if attention_mask is not None and attention_mask.device != self.device:
                attention_mask = attention_mask.to(self.device)
            
            outputs = self.model(
                input_values=input_values,
                attention_mask=attention_mask,
            )
            
            hidden_states = outputs.last_hidden_state
            
            if attention_mask is not None:
                # Handle sequence length mismatch between input and output
                seq_len = hidden_states.size(1)
                
                if attention_mask.size(1) != seq_len:
                    # Approximate the attention mask for the output sequence
                    reduction_factor = attention_mask.size(1) / seq_len
                    indices = torch.arange(seq_len, device=attention_mask.device) * reduction_factor
                    indices = indices.long().clamp(0, attention_mask.size(1) - 1)
                    attention_mask = attention_mask.gather(1, indices.unsqueeze(0).expand(attention_mask.size(0), -1))
                
                attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                hidden_states = hidden_states * attention_mask
                pooled = hidden_states.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = hidden_states.mean(dim=1)
            
            return pooled.cpu().numpy()


class Wav2Vec2RandomForestClassifier:
    """Random Forest classifier using Wav2Vec2 features."""
    
    def __init__(
        self,
        wav2vec2_model: str = "facebook/wav2vec2-base-960h",
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        device: str = "auto",
    ):
        """
        Initialize Random Forest classifier with Wav2Vec2 features.
        
        Args:
            wav2vec2_model: Pre-trained Wav2Vec2 model name
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required at a leaf
            random_state: Random state for reproducibility
            device: Device for feature extraction
        """
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            model_name=wav2vec2_model,
            device=device,
        )
        
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,  # Use all available cores
        )
        
        self.is_fitted = False
    
    def extract_features_from_dataloader(
        self, dataloader: torch.utils.data.DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and labels from a dataloader.
        
        Args:
            dataloader: PyTorch DataLoader
            
        Returns:
            Tuple of (features, labels)
        """
        all_features = []
        all_labels = []
        
        for batch in dataloader:
            input_values = batch["input_values"]
            labels = batch["label"].numpy()
            attention_mask = batch.get("attention_mask")
            
            # Extract features
            features = self.feature_extractor.extract_features(
                input_values=input_values,
                attention_mask=attention_mask,
            )
            
            all_features.append(features)
            all_labels.append(labels)
        
        if not all_features:
            # Return empty arrays with correct shape if no data
            # Use a reasonable feature dimension (768 for wav2vec2-base)
            return np.empty((0, 768)), np.array([])
        
        return np.vstack(all_features), np.concatenate(all_labels)
    
    def fit(
        self, 
        train_dataloader: torch.utils.data.DataLoader,
        verbose: bool = True,
    ) -> "Wav2Vec2RandomForestClassifier":
        """
        Train the Random Forest classifier.
        
        Args:
            train_dataloader: Training data loader
            verbose: Whether to print training progress
            
        Returns:
            Self for method chaining
        """
        if verbose:
            print("Extracting features from training data...")
        
        X_train, y_train = self.extract_features_from_dataloader(train_dataloader)
        
        if verbose:
            print(f"Training Random Forest on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        
        self.classifier.fit(X_train, y_train)
        self.is_fitted = True
        
        if verbose:
            print("Training completed!")
        
        return self
    
    def predict(self, dataloader: torch.utils.data.DataLoader) -> np.ndarray:
        """
        Make predictions on data.
        
        Args:
            dataloader: Data loader for prediction
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X, _ = self.extract_features_from_dataloader(dataloader)
        return self.classifier.predict(X)
    
    def predict_proba(self, dataloader: torch.utils.data.DataLoader) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            dataloader: Data loader for prediction
            
        Returns:
            Probability predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X, _ = self.extract_features_from_dataloader(dataloader)
        return self.classifier.predict_proba(X)
    
    def evaluate(
        self, 
        dataloader: torch.utils.data.DataLoader,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate the model on given data.
        
        Args:
            dataloader: Data loader for evaluation
            verbose: Whether to print results
            
        Returns:
            Dictionary with evaluation metrics
        """
        X, y_true = self.extract_features_from_dataloader(dataloader)
        y_pred = self.classifier.predict(X)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted"
        )
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
        
        if verbose:
            print(f"Evaluation Results:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def save(self, model_path: Union[str, Path]) -> None:
        """
        Save the trained model.
        
        Args:
            model_path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        joblib.dump(self.classifier, model_path)
        print(f"Model saved to {model_path}")
    
    def load(self, model_path: Union[str, Path]) -> "Wav2Vec2RandomForestClassifier":
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Self for method chaining
        """
        self.classifier = joblib.load(model_path)
        self.is_fitted = True
        print(f"Model loaded from {model_path}")
        return self