"""Data loading and preprocessing utilities."""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_classification
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Optional, List
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loading and preprocessing class."""
    
    def __init__(self, config):
        """Initialize data loader with configuration.
        
        Args:
            config: Configuration object with data settings.
        """
        self.config = config
        self.scaler = StandardScaler() if config.data.feature_scaling else None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.target_names = None
        self.feature_metadata = {}
    
    def load_dataset(self, dataset_name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Load dataset by name.
        
        Args:
            dataset_name: Name of the dataset to load.
            
        Returns:
            Tuple of (X, y, feature_names, target_names).
        """
        dataset_name = dataset_name or self.config.data.dataset_name
        
        if dataset_name == "iris":
            return self._load_iris()
        elif dataset_name == "wine":
            return self._load_wine()
        elif dataset_name == "breast_cancer":
            return self._load_breast_cancer()
        elif dataset_name == "synthetic":
            return self._generate_synthetic_data()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _load_iris(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Load Iris dataset."""
        data = load_iris()
        X = data.data
        y = data.target
        feature_names = data.feature_names
        target_names = data.target_names
        
        self.feature_metadata = {
            "feature_types": ["continuous"] * len(feature_names),
            "feature_ranges": [(float(X[:, i].min()), float(X[:, i].max())) for i in range(X.shape[1])],
            "sensitive_attributes": [],
            "monotonic_features": []
        }
        
        logger.info(f"Loaded Iris dataset: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, feature_names, target_names
    
    def _load_wine(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Load Wine dataset."""
        data = load_wine()
        X = data.data
        y = data.target
        feature_names = data.feature_names
        target_names = data.target_names
        
        self.feature_metadata = {
            "feature_types": ["continuous"] * len(feature_names),
            "feature_ranges": [(float(X[:, i].min()), float(X[:, i].max())) for i in range(X.shape[1])],
            "sensitive_attributes": [],
            "monotonic_features": []
        }
        
        logger.info(f"Loaded Wine dataset: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, feature_names, target_names
    
    def _load_breast_cancer(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Load Breast Cancer dataset."""
        data = load_breast_cancer()
        X = data.data
        y = data.target
        feature_names = data.feature_names
        target_names = data.target_names
        
        self.feature_metadata = {
            "feature_types": ["continuous"] * len(feature_names),
            "feature_ranges": [(float(X[:, i].min()), float(X[:, i].max())) for i in range(X.shape[1])],
            "sensitive_attributes": [],
            "monotonic_features": []
        }
        
        logger.info(f"Loaded Breast Cancer dataset: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, feature_names, target_names
    
    def _generate_synthetic_data(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Generate synthetic classification dataset."""
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_classes=3,
            random_state=self.config.data.random_state
        )
        
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        target_names = [f"class_{i}" for i in range(len(np.unique(y)))]
        
        self.feature_metadata = {
            "feature_types": ["continuous"] * len(feature_names),
            "feature_ranges": [(float(X[:, i].min()), float(X[:, i].max())) for i in range(X.shape[1])],
            "sensitive_attributes": [],
            "monotonic_features": []
        }
        
        logger.info(f"Generated synthetic dataset: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, feature_names, target_names
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the data according to configuration.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            
        Returns:
            Tuple of (X_processed, y_processed).
        """
        X_processed = X.copy()
        y_processed = y.copy()
        
        # Feature scaling
        if self.scaler is not None:
            X_processed = self.scaler.fit_transform(X_processed)
            logger.info("Applied feature scaling")
        
        # Label encoding
        y_processed = self.label_encoder.fit_transform(y_processed)
        
        return X_processed, y_processed
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.data.test_size,
            random_state=self.config.data.random_state,
            stratify=y
        )
        
        logger.info(f"Split data: train={X_train.shape[0]}, test={X_test.shape[0]}")
        return X_train, X_test, y_train, y_test
    
    def save_metadata(self, filepath: str) -> None:
        """Save feature metadata to JSON file.
        
        Args:
            filepath: Path to save metadata.
        """
        metadata = {
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "feature_metadata": self.feature_metadata,
            "config": {
                "dataset_name": self.config.data.dataset_name,
                "test_size": self.config.data.test_size,
                "random_state": self.config.data.random_state,
                "feature_scaling": self.config.data.feature_scaling
            }
        }
        
        # Use numpy's save function for better serialization
        np.save(filepath.replace('.json', '.npy'), metadata)
        
        logger.info(f"Saved metadata to {filepath.replace('.json', '.npy')}")
    
    def load_metadata(self, filepath: str) -> Dict[str, Any]:
        """Load feature metadata from JSON file.
        
        Args:
            filepath: Path to metadata file.
            
        Returns:
            Dictionary with metadata.
        """
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata["feature_names"]
        self.target_names = metadata["target_names"]
        self.feature_metadata = metadata["feature_metadata"]
        
        logger.info(f"Loaded metadata from {filepath}")
        return metadata
