"""Unit tests for SHAP XAI project."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.config import Config, load_config, save_config
from src.utils.device import set_seed, get_device
from src.data.loader import DataLoader
from src.models.manager import ModelManager
from src.explainers.shap_explainer import SHAPExplainer
from src.metrics.xai_metrics import XAIMetrics


class TestConfig:
    """Test configuration management."""
    
    def test_config_creation(self):
        """Test basic config creation."""
        config = Config()
        assert config.data.dataset_name == "iris"
        assert config.model.model_type == "random_forest"
        assert config.shap.explainer_type == "tree"
    
    def test_config_paths(self):
        """Test path creation."""
        config = Config()
        assert isinstance(config.data_dir, Path)
        assert isinstance(config.assets_dir, Path)
        assert config.data_dir.exists()
        assert config.assets_dir.exists()


class TestDataLoader:
    """Test data loading functionality."""
    
    def test_iris_loading(self):
        """Test Iris dataset loading."""
        config = Config()
        data_loader = DataLoader(config)
        
        X, y, feature_names, target_names = data_loader.load_dataset("iris")
        
        assert X.shape[1] == 4  # 4 features
        assert len(np.unique(y)) == 3  # 3 classes
        assert len(feature_names) == 4
        assert len(target_names) == 3
    
    def test_data_preprocessing(self):
        """Test data preprocessing."""
        config = Config()
        data_loader = DataLoader(config)
        
        X, y, _, _ = data_loader.load_dataset("iris")
        X_processed, y_processed = data_loader.preprocess_data(X, y)
        
        assert X_processed.shape == X.shape
        assert y_processed.shape == y.shape
        assert np.all(y_processed >= 0)  # Encoded labels should be non-negative
    
    def test_data_splitting(self):
        """Test data splitting."""
        config = Config()
        data_loader = DataLoader(config)
        
        X, y, _, _ = data_loader.load_dataset("iris")
        X_processed, y_processed = data_loader.preprocess_data(X, y)
        X_train, X_test, y_train, y_test = data_loader.split_data(X_processed, y_processed)
        
        assert len(X_train) + len(X_test) == len(X_processed)
        assert len(y_train) + len(y_test) == len(y_processed)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)


class TestModelManager:
    """Test model management."""
    
    def test_random_forest_creation(self):
        """Test Random Forest model creation."""
        config = Config()
        config.model.model_type = "random_forest"
        model_manager = ModelManager(config)
        
        model = model_manager.create_model()
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_model_training(self):
        """Test model training."""
        config = Config()
        model_manager = ModelManager(config)
        
        # Create dummy data
        X_train = np.random.randn(100, 4)
        y_train = np.random.randint(0, 3, 100)
        
        model = model_manager.train_model(X_train, y_train)
        assert model is not None
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        config = Config()
        model_manager = ModelManager(config)
        
        # Create dummy data
        X_train = np.random.randn(100, 4)
        y_train = np.random.randint(0, 3, 100)
        X_test = np.random.randn(30, 4)
        y_test = np.random.randint(0, 3, 30)
        
        model = model_manager.train_model(X_train, y_train)
        metrics = model_manager.evaluate_model(X_test, y_test)
        
        assert "accuracy" in metrics
        assert "predictions" in metrics
        assert "probabilities" in metrics
        assert 0 <= metrics["accuracy"] <= 1


class TestSHAPExplainer:
    """Test SHAP explainer functionality."""
    
    def test_explainer_creation(self):
        """Test explainer creation."""
        config = Config()
        shap_explainer = SHAPExplainer(config)
        
        # Mock model
        mock_model = Mock()
        mock_model.predict_proba = Mock(return_value=np.random.rand(10, 3))
        
        X_background = np.random.randn(50, 4)
        feature_names = ["f1", "f2", "f3", "f4"]
        
        explainer = shap_explainer.create_explainer(mock_model, X_background, feature_names)
        assert explainer is not None
    
    def test_feature_importance(self):
        """Test feature importance computation."""
        config = Config()
        shap_explainer = SHAPExplainer(config)
        
        # Mock SHAP values
        shap_values = np.random.randn(100, 4)
        shap_explainer.shap_values = shap_values
        shap_explainer.feature_names = ["f1", "f2", "f3", "f4"]
        
        importance = shap_explainer.get_feature_importance()
        
        assert len(importance) == 4
        assert all(isinstance(v, float) for v in importance.values())
        assert all(v >= 0 for v in importance.values())  # Importance should be non-negative


class TestXAIMetrics:
    """Test XAI evaluation metrics."""
    
    def test_faithfulness_deletion(self):
        """Test faithfulness deletion test."""
        config = Config()
        xai_metrics = XAIMetrics(config)
        
        # Mock model
        mock_model = Mock()
        mock_model.predict_proba = Mock(return_value=np.random.rand(10, 3))
        
        X_test = np.random.randn(10, 4)
        y_test = np.random.randint(0, 3, 10)
        shap_values = np.random.randn(10, 4)
        feature_names = ["f1", "f2", "f3", "f4"]
        
        metrics = xai_metrics.faithfulness_deletion(
            mock_model, X_test, y_test, shap_values, feature_names
        )
        
        assert "deletion_scores" in metrics
        assert "deletion_percentages" in metrics
        assert "original_accuracy" in metrics
    
    def test_stability_across_seeds(self):
        """Test stability across seeds."""
        config = Config()
        xai_metrics = XAIMetrics(config)
        
        # Create mock SHAP values for different seeds
        shap_values_list = [
            np.random.randn(10, 4),
            np.random.randn(10, 4),
            np.random.randn(10, 4)
        ]
        
        metrics = xai_metrics.stability_across_seeds(shap_values_list)
        
        assert "mean_spearman_correlation" in metrics
        assert "mean_kendall_tau" in metrics
        assert "n_comparisons" in metrics


class TestDeviceUtils:
    """Test device utilities."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # This is hard to test directly, but we can ensure it doesn't raise errors
        assert True
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device("auto")
        assert device is not None
        
        device = get_device("cpu")
        assert str(device) == "cpu"


if __name__ == "__main__":
    pytest.main([__file__])
