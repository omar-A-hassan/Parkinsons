# Parkinson's Disease Voice Detection 🎤🧠

A machine learning pipeline for detecting Parkinson's disease from voice recordings using Wav2Vec2 feature extraction and classification.

## Overview

This project implements a robust ML workflow for Parkinson's disease detection using voice data. It features:

- **Wav2Vec2 Feature Extraction**: Leverages pre-trained Wav2Vec2 models for audio feature extraction
- **Multiple Classification Approaches**: 
  - Fine-tuned Wav2Vec2 classifier with neural classification head
  - Random Forest classifier using Wav2Vec2 features as input
- **Production-Ready Pipeline**: Comprehensive preprocessing, training, and evaluation
- **Robust Testing**: Full test suite with unit and integration tests
- **CI/CD Ready**: GitHub Actions workflow for automated testing

## 📊 Dataset

The project works with voice recordings from:
- **HC_AH**: Healthy Control subjects (41 samples)
- **PD_AH**: Parkinson's Disease subjects (40 samples)

Audio characteristics:
- Sample rate: 8kHz (resampled to 16kHz at runtime)
- Format: WAV files
- Average duration: ~3.0-3.6 seconds

## 🚀 Quick Start

### Environment Setup

```bash
# Using conda (recommended)
conda env create -f environment.yml
conda activate parkinsons-voice

# Or using pip
pip install -e ".[dev,viz,experiment]"
```

### Data Preprocessing

```bash
# Create dataset metadata CSV (no audio resampling - done at runtime)
python scripts/preprocess_data.py
```

This will:
- Create dataset CSV with metadata (file paths, labels, durations)
- Process HC_AH and PD_AH directories
- Handle basic audio file validation
- Use configuration-driven preprocessing settings

### Training Models

#### Option 1: Wav2Vec2 Classifier (Fine-tuning)
```bash
# Uses config/training_config.yaml and config/model_config.yaml
python scripts/train_wav2vec.py
```

#### Option 2: Random Forest with Wav2Vec2 Features
```bash
# Uses config/random_forest_config.yaml
python scripts/train_random_forest.py
```

### Using Makefile

```bash
make install-dev      # Install with dev dependencies
make preprocess       # Create dataset metadata CSV
make train-wav2vec    # Train Wav2Vec2 classifier
make train-rf         # Train Random Forest classifier
make test             # Run all tests
make test-unit        # Run unit tests only
make test-integration # Run training smoke tests
make lint             # Run linting checks
make format           # Format code with black
make type-check       # Run mypy type checking
```

## 🏗️ Architecture

### Project Structure
```
parkinsons-voice/
├── src/parkinsons_voice/          # Main package
│   ├── data/                      # Data processing modules
│   │   ├── preprocessing.py       # Audio preprocessing
│   │   └── dataset.py            # PyTorch dataset classes
│   ├── models/                    # Model implementations
│   │   └── wav2vec2_classifier.py # Wav2Vec2-based models
│   └── train/                     # Training scripts
│       ├── train_wav2vec.py       # Neural classifier training
│       └── train_random_forest.py # Random Forest training
├── tests/                         # Test suite
├── scripts/                       # CLI scripts
├── data/                         # Raw data directory
└── models/                       # Saved models
```

### Data Pipeline

1. **Dataset Creation**: Create metadata CSV from HC_AH/PD_AH directories
2. **Audio Loading**: Load WAV files using librosa at runtime
3. **Preprocessing**: Resample to 16kHz, normalize, pad/truncate (runtime)
4. **Feature Extraction**: Wav2Vec2 processor for model input
5. **Dataset Loading**: PyTorch Dataset with train/validation splits

### Model Architectures

#### Wav2Vec2 Classifier
```
Wav2Vec2 Base Model (frozen/unfrozen)
    ↓
Global Average Pooling
    ↓
Dropout
    ↓
Linear(hidden_size → 256) + ReLU + Dropout
    ↓
Linear(256 → 2) [HC/PD classification]
```

#### Random Forest Pipeline
```
Audio Input
    ↓
Wav2Vec2 Feature Extractor (frozen)
    ↓
Global Average Pooling
    ↓
Random Forest Classifier
```

## 📝 Usage Examples

### Basic Dataset Usage
```python
from src.parkinsons_voice.data.dataset import ParkinsonsVoiceDataset

# Load dataset
dataset = ParkinsonsVoiceDataset('data/dataset_metadata.csv')
print(f"Dataset size: {len(dataset)}")
print(f"Class distribution: {dataset.get_class_distribution()}")

# Get a sample
sample = dataset[0]
print(f"Audio shape: {sample['input_values'].shape}")
print(f"Label: {sample['label']}")
```

### Feature Extraction
```python
from src.parkinsons_voice.models.wav2vec2_classifier import Wav2Vec2FeatureExtractor

extractor = Wav2Vec2FeatureExtractor()
features = extractor.extract_features(audio_tensor)
print(f"Features shape: {features.shape}")  # [batch_size, 768]
```

### Training Custom Models
```python
from src.parkinsons_voice.models.wav2vec2_classifier import Wav2Vec2RandomForestClassifier

# Initialize and train Random Forest
rf_model = Wav2Vec2RandomForestClassifier(n_estimators=100)
rf_model.fit(train_loader)

# Evaluate
metrics = rf_model.evaluate(val_loader)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_preprocessing.py -v
pytest tests/test_dataset.py -v
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=src/parkinsons_voice --cov-report=html
```

## 🔧 Configuration

All configuration is managed through YAML files in the `config/` directory:

- **`config/data_config.yaml`**: Data paths and preprocessing settings
- **`config/model_config.yaml`**: Wav2Vec2 model settings and audio parameters
- **`config/training_config.yaml`**: Training hyperparameters and device settings
- **`config/random_forest_config.yaml`**: Random Forest model parameters
- **`config/experiment_config.yaml`**: Experiment tracking and W&B settings

### Key Settings

**Audio Processing** (config/model_config.yaml):
- `target_sample_rate`: 16000
- `max_duration`: 10.0 seconds
- `normalize_audio`: True

**Training** (config/training_config.yaml):
- `batch_size`: 4
- `learning_rate`: 2e-5
- `epochs`: 10
- Device priority: MPS → CUDA → CPU

**Random Forest** (config/random_forest_config.yaml):
- `n_estimators`: 100
- Cross-validation and hyperparameter search options

## 📊 Model Performance

Expected performance metrics:
- **Validation Accuracy**: ~85-90%
- **F1-Score**: ~0.85-0.90
- **Training Time**: 
  - Wav2Vec2: ~30-60 minutes (10 epochs, GPU)
  - Random Forest: ~5-10 minutes (CPU)

## 🚀 Deployment

### GitHub Actions CI/CD

The project includes automated CI/CD with:
- Code quality checks (linting, formatting, type checking)
- Unit and integration testing
- Model training validation
- Multi-Python version support (3.9, 3.10)

### Model Serving

Models can be saved and loaded:
```python
# Save Wav2Vec2 model
torch.save(model.state_dict(), 'models/wav2vec2_classifier.pt')

# Save Random Forest
rf_model.save('models/random_forest.joblib')

# Load for inference
model.load_state_dict(torch.load('models/wav2vec2_classifier.pt'))
rf_model.load('models/random_forest.joblib')
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `make test`
5. Run linting: `make lint`
6. Submit a pull request

## 📚 References

- [Wav2Vec2 Paper](https://arxiv.org/abs/2006.11477)
- [PyTorch Audio Tutorial](https://pytorch.org/audio/stable/tutorials/speech_recognition_pipeline_tutorial.html)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♀️ Support

For questions or issues:
1. Check the [Issues](../../issues) page
2. Create a new issue with detailed description
3. Include system information and error logs