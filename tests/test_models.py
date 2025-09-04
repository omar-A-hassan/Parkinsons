"""Tests for model implementations."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import joblib

from src.parkinsons_voice.models.wav2vec2_classifier import (
    Wav2Vec2Classifier,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2RandomForestClassifier,
)


class TestWav2Vec2Classifier:
    """Test cases for Wav2Vec2Classifier."""
    
    @pytest.fixture
    def model(self):
        """Create a test model instance."""
        return Wav2Vec2Classifier(
            model_name="facebook/wav2vec2-base-960h",
            num_classes=2,
            dropout_rate=0.1,
            hidden_dim=128,
        )
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input data."""
        batch_size = 2
        seq_length = 16000  # 1 second at 16kHz
        input_values = torch.randn(batch_size, seq_length)
        attention_mask = torch.ones(batch_size, seq_length)
        return input_values, attention_mask
    
    def test_init(self):
        """Test model initialization."""
        model = Wav2Vec2Classifier(
            num_classes=3,
            dropout_rate=0.2,
            hidden_dim=256,
        )
        
        assert model.num_classes == 3
        assert model.wav2vec2 is not None
        assert model.classifier is not None
        assert isinstance(model.dropout, nn.Dropout)
    
    def test_init_with_freezing(self):
        """Test model initialization with component freezing."""
        model = Wav2Vec2Classifier(
            freeze_feature_extractor=True,
            freeze_transformer=True,
        )
        
        # Check that feature extractor parameters are frozen
        for param in model.wav2vec2.feature_extractor.parameters():
            assert not param.requires_grad
        
        # Check that transformer parameters are frozen
        for param in model.wav2vec2.encoder.parameters():
            assert not param.requires_grad
    
    def test_forward_basic(self, model, sample_input):
        """Test basic forward pass."""
        input_values, attention_mask = sample_input
        
        with torch.no_grad():
            logits = model(input_values, attention_mask)
        
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (2, 2)  # batch_size, num_classes
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
    
    def test_forward_without_attention_mask(self, model, sample_input):
        """Test forward pass without attention mask."""
        input_values, _ = sample_input
        
        with torch.no_grad():
            logits = model(input_values, attention_mask=None)
        
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (2, 2)
    
    def test_forward_single_sample(self, model):
        """Test forward pass with single sample."""
        input_values = torch.randn(1, 16000)
        
        with torch.no_grad():
            logits = model(input_values)
        
        assert logits.shape == (1, 2)
    
    def test_extract_features(self, model, sample_input):
        """Test feature extraction."""
        input_values, attention_mask = sample_input
        
        features = model.extract_features(input_values, attention_mask)
        
        assert isinstance(features, torch.Tensor)
        assert features.shape == (2, model.config.hidden_size)  # batch_size, hidden_size
        assert not torch.isnan(features).any()
    
    def test_extract_features_without_attention_mask(self, model, sample_input):
        """Test feature extraction without attention mask."""
        input_values, _ = sample_input
        
        features = model.extract_features(input_values)
        
        assert isinstance(features, torch.Tensor)
        assert features.shape == (2, model.config.hidden_size)
    
    def test_gradient_flow(self, model, sample_input):
        """Test that gradients flow properly during training."""
        input_values, attention_mask = sample_input
        targets = torch.tensor([0, 1])
        
        model.train()
        logits = model(input_values, attention_mask)
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, targets)
        loss.backward()
        
        # Check that gradients are computed for unfrozen parameters
        classifier_params = list(model.classifier.parameters())
        for param in classifier_params:
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()


class TestWav2Vec2FeatureExtractor:
    """Test cases for Wav2Vec2FeatureExtractor."""
    
    @pytest.fixture
    def extractor(self):
        """Create feature extractor instance."""
        return Wav2Vec2FeatureExtractor(device="cpu")
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input data."""
        batch_size = 2
        seq_length = 16000
        input_values = torch.randn(batch_size, seq_length)
        attention_mask = torch.ones(batch_size, seq_length)
        return input_values, attention_mask
    
    def test_init(self):
        """Test feature extractor initialization."""
        extractor = Wav2Vec2FeatureExtractor(device="cpu")
        
        assert extractor.model is not None
        assert extractor.device.type == "cpu"
    
    def test_init_auto_device(self):
        """Test automatic device selection."""
        extractor = Wav2Vec2FeatureExtractor(device="auto")
        
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert extractor.device.type == expected_device
    
    def test_extract_features(self, extractor, sample_input):
        """Test feature extraction."""
        input_values, attention_mask = sample_input
        
        features = extractor.extract_features(input_values, attention_mask)
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == 2  # batch_size
        assert features.shape[1] > 0  # hidden_size
        assert not np.isnan(features).any()
    
    def test_extract_features_without_attention_mask(self, extractor, sample_input):
        """Test feature extraction without attention mask."""
        input_values, _ = sample_input
        
        features = extractor.extract_features(input_values)
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == 2
    
    def test_extract_features_different_device(self, extractor):
        """Test feature extraction with tensors on different devices."""
        input_values = torch.randn(1, 16000)  # On CPU
        
        # Should work regardless of input device
        features = extractor.extract_features(input_values)
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == 1


class TestWav2Vec2RandomForestClassifier:
    """Test cases for Wav2Vec2RandomForestClassifier."""
    
    @pytest.fixture
    def mock_dataloader(self):
        """Create mock dataloader for testing."""
        batch_size = 4
        seq_length = 16000
        
        batch = {
            "input_values": torch.randn(batch_size, seq_length),
            "label": torch.tensor([0, 1, 0, 1]),
            "attention_mask": torch.ones(batch_size, seq_length),
        }
        
        mock_loader = Mock()
        # Use a lambda to create a new iterator each time
        mock_loader.__iter__ = Mock(side_effect=lambda: iter([batch]))
        return mock_loader
    
    @pytest.fixture
    def rf_classifier(self):
        """Create RandomForest classifier instance."""
        return Wav2Vec2RandomForestClassifier(
            n_estimators=10,  # Small for faster testing
            random_state=42,
            device="cpu",
        )
    
    def test_init(self):
        """Test RandomForest classifier initialization."""
        rf = Wav2Vec2RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=123,
        )
        
        assert rf.feature_extractor is not None
        assert rf.classifier is not None
        assert not rf.is_fitted
        assert rf.classifier.n_estimators == 50
        assert rf.classifier.max_depth == 10
        assert rf.classifier.random_state == 123
    
    @patch('src.parkinsons_voice.models.wav2vec2_classifier.Wav2Vec2FeatureExtractor')
    def test_extract_features_from_dataloader(self, mock_extractor_class, rf_classifier, mock_dataloader):
        """Test feature extraction from dataloader."""
        # Mock the feature extractor
        mock_extractor = Mock()
        mock_features = np.random.randn(4, 768)  # batch_size, hidden_size
        mock_extractor.extract_features.return_value = mock_features
        mock_extractor_class.return_value = mock_extractor
        
        # Create new classifier with mocked extractor
        rf = Wav2Vec2RandomForestClassifier(device="cpu")
        rf.feature_extractor = mock_extractor
        
        features, labels = rf.extract_features_from_dataloader(mock_dataloader)
        
        assert isinstance(features, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert features.shape == (4, 768)
        assert labels.shape == (4,)
        np.testing.assert_array_equal(labels, np.array([0, 1, 0, 1]))
    
    @patch('src.parkinsons_voice.models.wav2vec2_classifier.Wav2Vec2FeatureExtractor')
    def test_fit(self, mock_extractor_class, mock_dataloader):
        """Test model fitting."""
        # Mock the feature extractor
        mock_extractor = Mock()
        mock_features = np.random.randn(4, 768)
        mock_extractor.extract_features.return_value = mock_features
        mock_extractor_class.return_value = mock_extractor
        
        rf = Wav2Vec2RandomForestClassifier(n_estimators=10, device="cpu")
        rf.feature_extractor = mock_extractor
        
        # Fit the model
        rf.fit(mock_dataloader, verbose=False)
        
        assert rf.is_fitted
        # Check that the classifier was fitted
        assert hasattr(rf.classifier, 'n_features_in_')
    
    def test_fit_not_fitted_error(self, rf_classifier, mock_dataloader):
        """Test error when trying to predict without fitting."""
        with pytest.raises((ValueError, Exception), match="fitted|not fitted"):
            rf_classifier.predict(mock_dataloader)
        
        with pytest.raises((ValueError, Exception), match="fitted|not fitted"):
            rf_classifier.predict_proba(mock_dataloader)
        
        with pytest.raises((ValueError, Exception), match="fitted|not fitted"):
            rf_classifier.evaluate(mock_dataloader)
    
    @patch('src.parkinsons_voice.models.wav2vec2_classifier.Wav2Vec2FeatureExtractor')
    def test_predict_and_evaluate(self, mock_extractor_class, mock_dataloader):
        """Test prediction and evaluation."""
        # Mock the feature extractor with consistent data
        mock_extractor = Mock()
        mock_features = np.random.randn(4, 768)
        
        def side_effect(*args, **kwargs):
            return mock_features
        
        mock_extractor.extract_features.side_effect = side_effect
        mock_extractor_class.return_value = mock_extractor
        
        rf = Wav2Vec2RandomForestClassifier(n_estimators=10, device="cpu")
        rf.feature_extractor = mock_extractor
        
        # Fit the model
        rf.fit(mock_dataloader, verbose=False)
        
        # Test prediction
        predictions = rf.predict(mock_dataloader)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 4
        assert all(pred in [0, 1] for pred in predictions)
        
        # Test probability prediction
        probabilities = rf.predict_proba(mock_dataloader)
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (4, 2)  # batch_size, num_classes
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
        
        # Test evaluation
        metrics = rf.evaluate(mock_dataloader, verbose=False)
        assert isinstance(metrics, dict)
        expected_metrics = {"accuracy", "precision", "recall", "f1_score"}
        assert set(metrics.keys()) == expected_metrics
        for metric_value in metrics.values():
            assert 0.0 <= metric_value <= 1.0
    
    def test_save_load_not_fitted(self, rf_classifier, tmp_path):
        """Test error when saving unfitted model."""
        model_path = tmp_path / "model.joblib"
        
        with pytest.raises((ValueError, Exception), match="fitted|not fitted"):
            rf_classifier.save(model_path)
    
    @patch('src.parkinsons_voice.models.wav2vec2_classifier.Wav2Vec2FeatureExtractor')
    def test_save_load_model(self, mock_extractor_class, mock_dataloader, tmp_path):
        """Test model saving and loading."""
        # Mock the feature extractor
        mock_extractor = Mock()
        mock_features = np.random.randn(4, 768)
        
        def side_effect(*args, **kwargs):
            return mock_features
        
        mock_extractor.extract_features.side_effect = side_effect
        mock_extractor_class.return_value = mock_extractor
        
        rf = Wav2Vec2RandomForestClassifier(n_estimators=10, device="cpu")
        rf.feature_extractor = mock_extractor
        
        # Fit and save the model
        rf.fit(mock_dataloader, verbose=False)
        model_path = tmp_path / "test_model.joblib"
        rf.save(model_path)
        
        assert model_path.exists()
        
        # Create new instance and load
        rf_new = Wav2Vec2RandomForestClassifier(device="cpu")
        rf_new.feature_extractor = mock_extractor  # Set mock extractor
        rf_new.load(model_path)
        
        assert rf_new.is_fitted
        
        # Test that loaded model can make predictions
        predictions = rf_new.predict(mock_dataloader)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 4


class TestModelIntegration:
    """Integration tests for models."""
    
    def test_wav2vec2_classifier_training_step(self):
        """Test a single training step with Wav2Vec2Classifier."""
        model = Wav2Vec2Classifier(num_classes=2, hidden_dim=64)  # Smaller for speed
        
        # Create sample batch
        batch_size = 2
        input_values = torch.randn(batch_size, 8000)  # Shorter sequence
        labels = torch.tensor([0, 1])
        
        # Forward pass
        model.train()
        logits = model(input_values)
        
        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that loss is computed and finite
        assert torch.isfinite(loss)
        assert loss.item() > 0
    
    def test_feature_consistency(self):
        """Test that feature extraction is consistent."""
        extractor = Wav2Vec2FeatureExtractor(device="cpu")
        
        # Same input should produce same features
        input_values = torch.randn(1, 16000)
        
        features1 = extractor.extract_features(input_values)
        features2 = extractor.extract_features(input_values)
        
        np.testing.assert_array_almost_equal(features1, features2, decimal=6)
    
    def test_model_modes(self):
        """Test model train/eval modes."""
        model = Wav2Vec2Classifier(num_classes=2)
        input_values = torch.randn(1, 8000)
        
        # Test eval mode (should be deterministic due to dropout)
        model.eval()
        with torch.no_grad():
            output1 = model(input_values)
            output2 = model(input_values)
        
        torch.testing.assert_close(output1, output2, rtol=1e-6, atol=1e-6)
        
        # Test train mode
        model.train()
        output_train = model(input_values)
        
        # Outputs should be different due to dropout (with high probability)
        # We just check that forward pass works
        assert output_train.shape == output1.shape