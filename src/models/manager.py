"""Model training and management utilities."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, Any, Optional, Tuple, Union
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelManager:
    """Model training and management class."""
    
    def __init__(self, config):
        """Initialize model manager with configuration.
        
        Args:
            config: Configuration object with model settings.
        """
        self.config = config
        self.model = None
        self.model_type = config.model.model_type
        self.device = None
        
    def create_model(self) -> Union[RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, nn.Module]:
        """Create model based on configuration.
        
        Returns:
            Trained model object.
        """
        if self.model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=self.config.model.n_estimators,
                max_depth=self.config.model.max_depth,
                min_samples_split=self.config.model.min_samples_split,
                min_samples_leaf=self.config.model.min_samples_leaf,
                random_state=self.config.model.random_state
            )
        elif self.model_type == "logistic_regression":
            model = LogisticRegression(
                random_state=self.config.model.random_state,
                max_iter=1000
            )
        elif self.model_type == "decision_tree":
            model = DecisionTreeClassifier(
                max_depth=self.config.model.max_depth,
                min_samples_split=self.config.model.min_samples_split,
                min_samples_leaf=self.config.model.min_samples_leaf,
                random_state=self.config.model.random_state
            )
        elif self.model_type == "neural_network":
            model = self._create_neural_network()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model = model
        logger.info(f"Created {self.model_type} model")
        return model
    
    def _create_neural_network(self) -> nn.Module:
        """Create neural network model.
        
        Returns:
            PyTorch neural network model.
        """
        from .neural_network import SimpleNeuralNetwork
        
        # This will be defined in a separate file
        model = SimpleNeuralNetwork(
            input_size=4,  # Will be updated based on data
            hidden_sizes=[64, 32],
            output_size=3,  # Will be updated based on data
            dropout_rate=0.2
        )
        
        self.device = torch.device(self.config.model.device)
        model = model.to(self.device)
        
        return model
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Train the model.
        
        Args:
            X_train: Training features.
            y_train: Training targets.
            
        Returns:
            Trained model.
        """
        if self.model is None:
            self.create_model()
        
        if self.model_type == "neural_network":
            return self._train_neural_network(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)
            logger.info(f"Trained {self.model_type} model")
            return self.model
    
    def _train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray) -> nn.Module:
        """Train neural network model.
        
        Args:
            X_train: Training features.
            y_train: Training targets.
            
        Returns:
            Trained neural network model.
        """
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.LongTensor(y_train).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        self.model.train()
        for epoch in range(100):  # Will be configurable
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        logger.info("Trained neural network model")
        return self.model
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance.
        
        Args:
            X_test: Test features.
            y_test: Test targets.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        if self.model_type == "neural_network":
            return self._evaluate_neural_network(X_test, y_test)
        else:
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            metrics = {
                "accuracy": accuracy,
                "predictions": y_pred,
                "probabilities": y_pred_proba,
                "classification_report": classification_report(y_test, y_pred, output_dict=True),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
            }
            
            logger.info(f"Model accuracy: {accuracy:.4f}")
            return metrics
    
    def _evaluate_neural_network(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate neural network model.
        
        Args:
            X_test: Test features.
            y_test: Test targets.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            # Convert back to numpy
            y_pred = predictions.cpu().numpy()
            y_pred_proba = probabilities.cpu().numpy()
        
        accuracy = accuracy_score(y_test, y_pred)
        
        metrics = {
            "accuracy": accuracy,
            "predictions": y_pred,
            "probabilities": y_pred_proba,
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }
        
        logger.info(f"Neural network accuracy: {accuracy:.4f}")
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """Save model to file.
        
        Args:
            filepath: Path to save the model.
        """
        if self.model_type == "neural_network":
            torch.save(self.model.state_dict(), filepath)
        else:
            joblib.dump(self.model, filepath)
        
        logger.info(f"Saved model to {filepath}")
    
    def load_model(self, filepath: str) -> Any:
        """Load model from file.
        
        Args:
            filepath: Path to load the model from.
            
        Returns:
            Loaded model.
        """
        if self.model_type == "neural_network":
            if self.model is None:
                self.create_model()
            self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        else:
            self.model = joblib.load(filepath)
        
        logger.info(f"Loaded model from {filepath}")
        return self.model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dictionary with model information.
        """
        info = {
            "model_type": self.model_type,
            "device": str(self.device) if self.device else None,
            "is_trained": self.model is not None
        }
        
        if self.model is not None and hasattr(self.model, 'get_params'):
            info["parameters"] = self.model.get_params()
        
        return info
