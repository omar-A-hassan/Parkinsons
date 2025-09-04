"""Training script for Random Forest classifier with Wav2Vec2 features."""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np

from ..data.dataset import ParkinsonsVoiceDataset, create_train_val_split, create_dataloaders
from ..models.wav2vec2_classifier import Wav2Vec2RandomForestClassifier
from ..utils.config import config_loader, TrainingConfig, ModelConfig, get_device_from_config


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def hyperparameter_search(
    model: Wav2Vec2RandomForestClassifier,
    train_loader,
    param_grid: Dict[str, Any],
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """
    Perform hyperparameter search for Random Forest.
    
    Args:
        model: Wav2Vec2RandomForestClassifier instance
        train_loader: Training data loader
        param_grid: Parameter grid for search
        cv_folds: Number of cross-validation folds
        
    Returns:
        Best parameters and scores
    """
    logger.info("Extracting features for hyperparameter search...")
    X_train, y_train = model.extract_features_from_dataloader(train_loader)
    
    logger.info(f"Performing grid search with {cv_folds}-fold CV...")
    
    grid_search = GridSearchCV(
        model.classifier,
        param_grid,
        cv=cv_folds,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    
    grid_search.fit(X_train, y_train)
    
    return {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "cv_results": grid_search.cv_results_,
    }


def main():
    """Main training function for Random Forest classifier."""
    parser = argparse.ArgumentParser(
        description="Train Random Forest classifier with Wav2Vec2 features for Parkinson's detection"
    )
    parser.add_argument("--config_dir", type=str, default="config",
                       help="Directory containing configuration files")
    parser.add_argument("--data_csv", type=str, default=None,
                       help="Path to dataset CSV file (overrides config)")
    parser.add_argument("--hyperparameter_search", action="store_true",
                       help="Perform hyperparameter search")
    parser.add_argument("--cv_folds", type=int, default=5,
                       help="Number of cross-validation folds for hyperparameter search")
    
    args = parser.parse_args()
    
    # Load configurations
    global config_loader
    config_loader = config_loader.__class__(args.config_dir)
    
    training_config = TrainingConfig.from_config(config_loader)
    model_config = ModelConfig.from_config(config_loader)
    
    # Get config values
    data_csv = args.data_csv or config_loader.get("data_config", "paths.dataset_csv", "data/dataset_metadata.csv")
    save_dir = config_loader.get("training_config", "saving.save_dir", "models")
    device = get_device_from_config(config_loader)
    
    # Random Forest specific config
    rf_config = config_loader.get("model_config", "random_forest", {})
    
    # Set up paths
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
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
        num_workers=training_config.num_workers
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Initialize Random Forest classifier
    rf_model = Wav2Vec2RandomForestClassifier(
        wav2vec2_model=model_config.model_name,
        n_estimators=rf_config.get("n_estimators", 100),
        max_depth=rf_config.get("max_depth", None),
        min_samples_split=rf_config.get("min_samples_split", 2),
        min_samples_leaf=rf_config.get("min_samples_leaf", 1),
        random_state=rf_config.get("random_state", 42),
        device=device,
    )
    
    # Hyperparameter search if requested
    if args.hyperparameter_search:
        logger.info("Starting hyperparameter search...")
        
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        
        search_results = hyperparameter_search(
            rf_model, train_loader, param_grid, args.cv_folds
        )
        
        logger.info(f"Best parameters: {search_results['best_params']}")
        logger.info(f"Best CV score: {search_results['best_score']:.4f}")
        
        # Update model with best parameters
        best_params = search_results["best_params"]
        rf_model.classifier.set_params(**best_params)
        
        # Save search results
        search_results_path = save_dir / "hyperparameter_search_results.json"
        with open(search_results_path, "w") as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_results = {}
            for key, value in search_results.items():
                if key == "cv_results":
                    # Skip cv_results as it's too large and not easily serializable
                    continue
                elif isinstance(value, np.integer):
                    serializable_results[key] = int(value)
                elif isinstance(value, np.floating):
                    serializable_results[key] = float(value)
                else:
                    serializable_results[key] = value
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Hyperparameter search results saved to {search_results_path}")
    
    # Train the model
    logger.info("Training Random Forest classifier...")
    rf_model.fit(train_loader, verbose=True)
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_metrics = rf_model.evaluate(val_loader, verbose=True)
    
    # Get detailed classification report
    logger.info("Generating detailed classification report...")
    X_val, y_val = rf_model.extract_features_from_dataloader(val_loader)
    y_pred = rf_model.classifier.predict(X_val)
    y_proba = rf_model.classifier.predict_proba(X_val)
    
    # Classification report
    class_report = classification_report(y_val, y_pred, target_names=["HC", "PD"])
    logger.info(f"Classification Report:\n{class_report}")
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_val, y_pred)
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    
    # Feature importance
    feature_importance = rf_model.classifier.feature_importances_
    logger.info(f"Top 10 feature importances:")
    top_features = np.argsort(feature_importance)[-10:][::-1]
    for i, idx in enumerate(top_features):
        logger.info(f"  Feature {idx}: {feature_importance[idx]:.4f}")
    
    # Save the trained model
    model_path = save_dir / "random_forest_model.joblib"
    rf_model.save(model_path)
    
    # Save evaluation results
    results = {
        "validation_metrics": val_metrics,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix.tolist(),
        "feature_importance_stats": {
            "mean": float(np.mean(feature_importance)),
            "std": float(np.std(feature_importance)),
            "max": float(np.max(feature_importance)),
            "min": float(np.min(feature_importance)),
        },
        "model_params": rf_model.classifier.get_params(),
    }
    
    results_path = save_dir / "random_forest_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    logger.info("Training completed successfully!")
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Model: Random Forest with Wav2Vec2 features")
    print(f"Dataset: {len(dataset)} samples ({len(train_dataset)} train, {len(val_dataset)} val)")
    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Validation F1-Score: {val_metrics['f1_score']:.4f}")
    print(f"Model saved to: {model_path}")
    print("="*50)


if __name__ == "__main__":
    main()