"""Integration smoke tests for training pipelines."""

import pytest
import tempfile
import shutil
from pathlib import Path
import json
import joblib
import torch
import numpy as np
import pandas as pd
import soundfile as sf

from src.parkinsons_voice.train.train_random_forest import main as train_rf_main
from src.parkinsons_voice.train.train_wav2vec import main as train_wav2vec_main
from src.parkinsons_voice.models.wav2vec2_classifier import (
    Wav2Vec2RandomForestClassifier, 
    Wav2Vec2Classifier
)


@pytest.fixture
def minimal_training_setup(tmp_path):
    """Create minimal setup for training smoke tests."""
    # Create directories
    data_dir = tmp_path / "data"
    hc_dir = data_dir / "HC_AH"
    pd_dir = data_dir / "PD_AH"
    config_dir = tmp_path / "config"
    models_dir = tmp_path / "models"
    
    for dir_path in [data_dir, hc_dir, pd_dir, config_dir, models_dir]:
        dir_path.mkdir(parents=True)
    
    # Create minimal audio files (0.5 seconds each)
    sample_rate = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create 4 HC samples
    for i in range(4):
        waveform = 0.3 * np.sin(2 * np.pi * (440 + i * 50) * t)  # Different frequencies
        sf.write(hc_dir / f"hc_sample_{i}.wav", waveform, sample_rate)
    
    # Create 4 PD samples  
    for i in range(4):
        waveform = 0.3 * np.sin(2 * np.pi * (220 + i * 30) * t)  # Different frequencies
        sf.write(pd_dir / f"pd_sample_{i}.wav", waveform, sample_rate)
    
    # Create dataset CSV
    data = []
    for file_path in hc_dir.glob("*.wav"):
        data.append({
            "file_path": str(file_path),
            "file_name": file_path.name,
            "label": 0,
            "class_name": "HC",
            "duration": duration,
            "original_sample_rate": sample_rate,
        })
    
    for file_path in pd_dir.glob("*.wav"):
        data.append({
            "file_path": str(file_path),
            "file_name": file_path.name, 
            "label": 1,
            "class_name": "PD",
            "duration": duration,
            "original_sample_rate": sample_rate,
        })
    
    df = pd.DataFrame(data)
    dataset_csv = data_dir / "dataset_metadata.csv"
    df.to_csv(dataset_csv, index=False)
    
    # Create minimal config files
    training_config = {
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "val_split": 0.3,
            "stratify": True,
        },
        "optimization": {
            "weight_decay": 1e-5,
            "warmup_ratio": 0.1,
            "grad_clip_max_norm": 1.0,
        },
        "device": {"priority": ["cpu"]},
        "saving": {"save_dir": str(models_dir)},
        "logging": {"log_every": 1, "use_wandb": False},
        "dataloader": {"num_workers": 0, "pin_memory": False},
    }
    
    model_config = {
        "model": {
            "model_name": "facebook/wav2vec2-base-960h",
            "num_classes": 2,
            "freeze_feature_extractor": False,
            "freeze_transformer": False,
            "dropout_rate": 0.1,
            "hidden_dim": 128,
        },
        "audio": {
            "target_sample_rate": 16000,
            "max_duration": 1.0,  # Short for smoke test
            "normalize_audio": True,
        },
        "random_forest": {
            "n_estimators": 10,  # Small for smoke test
            "random_state": 42,
        },
    }
    
    data_config = {
        "paths": {
            "data_dir": str(data_dir),
            "hc_dir": str(hc_dir),
            "pd_dir": str(pd_dir),
            "dataset_csv": str(dataset_csv),
        }
    }
    
    experiment_config = {
        "experiment": {
            "name": "smoke_test",
            "wandb_project": "test-project",
        }
    }
    
    # Save config files
    import yaml
    with open(config_dir / "training_config.yaml", "w") as f:
        yaml.dump(training_config, f)
    with open(config_dir / "model_config.yaml", "w") as f:
        yaml.dump(model_config, f)
    with open(config_dir / "data_config.yaml", "w") as f:
        yaml.dump(data_config, f)
    with open(config_dir / "experiment_config.yaml", "w") as f:
        yaml.dump(experiment_config, f)
    
    return {
        "data_dir": data_dir,
        "config_dir": config_dir,
        "models_dir": models_dir,
        "dataset_csv": dataset_csv,
    }


class TestRandomForestTrainingIntegration:
    """Integration tests for Random Forest training pipeline."""
    
    def test_rf_training_smoke_test(self, minimal_training_setup, monkeypatch):
        """Smoke test: RF training completes without errors."""
        setup = minimal_training_setup
        
        # Mock command line arguments
        import sys
        test_args = [
            "train_random_forest.py",
            "--config_dir", str(setup["config_dir"]),
            "--data_csv", str(setup["dataset_csv"]),
        ]
        monkeypatch.setattr(sys, "argv", test_args)
        
        # Run training
        train_rf_main()
        
        # Verify outputs
        models_dir = setup["models_dir"]
        assert (models_dir / "random_forest_model.joblib").exists()
        assert (models_dir / "random_forest_results.json").exists()
        
        # Verify model can be loaded (joblib loads the sklearn classifier directly)
        classifier = joblib.load(models_dir / "random_forest_model.joblib")
        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(classifier, RandomForestClassifier)
        assert classifier.n_estimators == 10
        
        # Verify we can recreate the wrapper class
        rf_model = Wav2Vec2RandomForestClassifier(n_estimators=10)
        rf_model.classifier = classifier
        assert hasattr(rf_model, "feature_extractor")
        assert hasattr(rf_model, "classifier")
        
        # Verify results file
        with open(models_dir / "random_forest_results.json") as f:
            results = json.load(f)
        assert "validation_metrics" in results
        assert "accuracy" in results["validation_metrics"]
        assert 0 <= results["validation_metrics"]["accuracy"] <= 1
    
    def test_rf_hyperparameter_search_smoke_test(self, minimal_training_setup, monkeypatch):
        """Smoke test: RF hyperparameter search completes."""
        setup = minimal_training_setup
        
        import sys
        test_args = [
            "train_random_forest.py",
            "--config_dir", str(setup["config_dir"]),
            "--data_csv", str(setup["dataset_csv"]),
            "--hyperparameter_search",
            "--cv_folds", "2",  # Minimal for smoke test
        ]
        monkeypatch.setattr(sys, "argv", test_args)
        
        # Run training with hyperparameter search
        train_rf_main()
        
        # Verify hyperparameter search results
        models_dir = setup["models_dir"]
        assert (models_dir / "hyperparameter_search_results.json").exists()
        
        with open(models_dir / "hyperparameter_search_results.json") as f:
            search_results = json.load(f)
        assert "best_params" in search_results
        assert "best_score" in search_results


class TestWav2Vec2TrainingIntegration:
    """Integration tests for Wav2Vec2 training pipeline."""
    
    def test_wav2vec2_training_smoke_test(self, minimal_training_setup, monkeypatch):
        """Smoke test: Wav2Vec2 training completes without errors."""
        setup = minimal_training_setup
        
        import sys
        test_args = [
            "train_wav2vec.py",
            "--config_dir", str(setup["config_dir"]),
            "--data_csv", str(setup["dataset_csv"]),
        ]
        monkeypatch.setattr(sys, "argv", test_args)
        
        # Run training
        train_wav2vec_main()
        
        # Verify outputs
        models_dir = setup["models_dir"]
        assert (models_dir / "best_model.pt").exists()
        assert (models_dir / "latest_model.pt").exists()
        assert (models_dir / "training_history.json").exists()
        
        # Verify model checkpoint can be loaded
        checkpoint = torch.load(models_dir / "best_model.pt", map_location="cpu")
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "metrics" in checkpoint
        assert "epoch" in checkpoint
        
        # Verify training history
        with open(models_dir / "training_history.json") as f:
            history = json.load(f)
        assert "train_loss" in history
        assert "val_loss" in history
        assert "val_acc" in history
        assert len(history["train_loss"]) == 2  # 2 epochs
    
    def test_wav2vec2_frozen_feature_extractor(self, minimal_training_setup, monkeypatch):
        """Test Wav2Vec2 training with frozen feature extractor."""
        setup = minimal_training_setup
        
        # Update config to freeze feature extractor
        config_path = setup["config_dir"] / "model_config.yaml"
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        config["model"]["freeze_feature_extractor"] = True
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        import sys
        test_args = [
            "train_wav2vec.py",
            "--config_dir", str(setup["config_dir"]),
            "--data_csv", str(setup["dataset_csv"]),
        ]
        monkeypatch.setattr(sys, "argv", test_args)
        
        # Run training
        train_wav2vec_main()
        
        # Should complete without errors
        models_dir = setup["models_dir"]
        assert (models_dir / "best_model.pt").exists()


class TestTrainingCompatibility:
    """Test compatibility between training approaches."""
    
    def test_same_dataset_different_models(self, minimal_training_setup, monkeypatch):
        """Test that both models can train on same dataset."""
        setup = minimal_training_setup
        
        # Train Random Forest
        import sys
        rf_args = [
            "train_random_forest.py",
            "--config_dir", str(setup["config_dir"]),
            "--data_csv", str(setup["dataset_csv"]),
        ]
        monkeypatch.setattr(sys, "argv", rf_args)
        train_rf_main()
        
        # Verify RF model exists
        rf_model_path = setup["models_dir"] / "random_forest_model.joblib"
        assert rf_model_path.exists()
        
        # Train Wav2Vec2 (save to different location to avoid conflict)
        wav2vec_dir = setup["models_dir"] / "wav2vec"
        wav2vec_dir.mkdir()
        
        # Update config for wav2vec
        config_path = setup["config_dir"] / "training_config.yaml"
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        config["saving"]["save_dir"] = str(wav2vec_dir)
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        wav2vec_args = [
            "train_wav2vec.py", 
            "--config_dir", str(setup["config_dir"]),
            "--data_csv", str(setup["dataset_csv"]),
        ]
        monkeypatch.setattr(sys, "argv", wav2vec_args)
        train_wav2vec_main()
        
        # Verify Wav2Vec2 model exists
        assert (wav2vec_dir / "best_model.pt").exists()
        
        # Both models should exist
        assert rf_model_path.exists()
        assert (wav2vec_dir / "best_model.pt").exists()


class TestTrainingErrorHandling:
    """Test error handling in training pipelines."""
    
    def test_invalid_dataset_path(self, monkeypatch):
        """Test training with invalid dataset path."""
        import sys
        test_args = [
            "train_random_forest.py",
            "--data_csv", "nonexistent.csv",
        ]
        monkeypatch.setattr(sys, "argv", test_args)
        
        # Should raise appropriate error
        with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError)):
            train_rf_main()