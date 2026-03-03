"""SHAP-based explainers for model interpretation."""

import numpy as np
import pandas as pd
import shap
import torch
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """Comprehensive SHAP explainer for various model types."""
    
    def __init__(self, config):
        """Initialize SHAP explainer with configuration.
        
        Args:
            config: Configuration object with SHAP settings.
        """
        self.config = config
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        self.model = None
        
    def create_explainer(self, model: Any, X_background: np.ndarray, feature_names: Optional[List[str]] = None) -> Any:
        """Create SHAP explainer based on model type.
        
        Args:
            model: Trained model to explain.
            X_background: Background dataset for explainer.
            feature_names: Names of features.
            
        Returns:
            SHAP explainer object.
        """
        self.model = model
        self.feature_names = feature_names
        
        explainer_type = self.config.shap.explainer_type
        
        if explainer_type == "tree":
            self.explainer = shap.TreeExplainer(model)
        elif explainer_type == "kernel":
            self.explainer = shap.KernelExplainer(
                model.predict_proba,
                X_background[:self.config.shap.background_samples],
                feature_names=feature_names
            )
        elif explainer_type == "deep":
            if hasattr(model, 'predict_proba'):
                # Wrap sklearn model for deep explainer
                def model_fn(x):
                    return model.predict_proba(x)
                self.explainer = shap.DeepExplainer(model_fn, X_background[:self.config.shap.background_samples])
            else:
                # PyTorch model
                self.explainer = shap.DeepExplainer(model, X_background[:self.config.shap.background_samples])
        elif explainer_type == "linear":
            self.explainer = shap.LinearExplainer(model, X_background)
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")
        
        logger.info(f"Created {explainer_type} SHAP explainer")
        return self.explainer
    
    def compute_shap_values(self, X: np.ndarray, max_samples: Optional[int] = None) -> np.ndarray:
        """Compute SHAP values for given data.
        
        Args:
            X: Data to compute SHAP values for.
            max_samples: Maximum number of samples to process.
            
        Returns:
            SHAP values array.
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")
        
        max_samples = max_samples or self.config.shap.max_samples
        
        if len(X) > max_samples:
            # Sample data for efficiency
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        try:
            if self.config.shap.explainer_type == "kernel":
                self.shap_values = self.explainer.shap_values(X_sample)
            else:
                self.shap_values = self.explainer.shap_values(X_sample)
            
            logger.info(f"Computed SHAP values for {len(X_sample)} samples")
            return self.shap_values
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            raise
    
    def get_feature_importance(self, class_idx: Optional[int] = None) -> Dict[str, float]:
        """Get global feature importance from SHAP values.
        
        Args:
            class_idx: Class index for multi-class problems.
            
        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values first.")
        
        if isinstance(self.shap_values, list):
            # Multi-class case - average across all classes
            if class_idx is None:
                # Average across all classes
                shap_vals = np.mean([np.abs(sv) for sv in self.shap_values], axis=0)
            else:
                shap_vals = self.shap_values[class_idx]
        else:
            shap_vals = self.shap_values
        
        # Compute mean absolute SHAP values
        importance_scores = np.mean(np.abs(shap_vals), axis=0)
        
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance_scores))]
        else:
            feature_names = self.feature_names
        
        return dict(zip(feature_names, importance_scores.tolist()))
    
    def explain_prediction(self, instance: np.ndarray, class_idx: Optional[int] = None) -> Dict[str, Any]:
        """Explain a single prediction.
        
        Args:
            instance: Single instance to explain.
            class_idx: Class index for multi-class problems.
            
        Returns:
            Dictionary with explanation details.
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")
        
        # Reshape for single instance
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)
        
        try:
            shap_values = self.explainer.shap_values(instance)
            
            if isinstance(shap_values, list):
                if class_idx is None:
                    class_idx = 0
                shap_vals = shap_values[class_idx][0]  # First instance
            else:
                shap_vals = shap_values[0]  # First instance
            
            # Get prediction
            if hasattr(self.model, 'predict_proba'):
                prediction = self.model.predict_proba(instance)[0]
            else:
                prediction = self.model.predict(instance)[0]
            
            explanation = {
                "shap_values": shap_vals.tolist(),
                "prediction": prediction.tolist() if isinstance(prediction, np.ndarray) else prediction,
                "feature_names": self.feature_names,
                "instance": instance[0].tolist()
            }
            
            return explanation
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            raise
    
    def plot_summary(self, class_idx: Optional[int] = None, max_display: int = 10, save_path: Optional[str] = None) -> None:
        """Plot SHAP summary plot.
        
        Args:
            class_idx: Class index for multi-class problems.
            max_display: Maximum number of features to display.
            save_path: Path to save the plot.
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values first.")
        
        plt.figure(figsize=self.config.visualization.figure_size)
        
        if isinstance(self.shap_values, list):
            if class_idx is None:
                class_idx = 0
            shap_vals = self.shap_values[class_idx]
        else:
            shap_vals = self.shap_values
        
        shap.summary_plot(
            shap_vals,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        plt.title(f"SHAP Summary Plot (Class {class_idx if class_idx is not None else 0})")
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.visualization.dpi, bbox_inches='tight')
            logger.info(f"Saved summary plot to {save_path}")
        
        plt.show()
    
    def plot_waterfall(self, instance_idx: int, class_idx: Optional[int] = None, save_path: Optional[str] = None) -> None:
        """Plot SHAP waterfall plot for a single instance.
        
        Args:
            instance_idx: Index of instance to explain.
            class_idx: Class index for multi-class problems.
            save_path: Path to save the plot.
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values first.")
        
        if isinstance(self.shap_values, list):
            if class_idx is None:
                class_idx = 0
            shap_vals = self.shap_values[class_idx]
        else:
            shap_vals = self.shap_values
        
        plt.figure(figsize=self.config.visualization.figure_size)
        
        # For multi-class, use the first class or specified class
        if isinstance(self.shap_values, list):
            if class_idx is None:
                class_idx = 0
            shap_vals = self.shap_values[class_idx]
        else:
            shap_vals = self.shap_values
        
        # Create a simple bar plot instead of waterfall for multi-class
        values = shap_vals[instance_idx]
        if len(values.shape) > 1:
            # Multi-class case - take the first class
            values = values[0] if values.shape[0] == 1 else values[:, 0]
        
        plt.bar(range(len(values)), values)
        plt.xticks(range(len(values)), self.feature_names, rotation=45)
        plt.xlabel('Features')
        plt.ylabel('SHAP Value')
        plt.title(f'SHAP Values for Instance {instance_idx} (Class {class_idx})')
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.visualization.dpi, bbox_inches='tight')
            logger.info(f"Saved waterfall plot to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, class_idx: Optional[int] = None, save_path: Optional[str] = None) -> None:
        """Plot feature importance bar chart.
        
        Args:
            class_idx: Class index for multi-class problems.
            save_path: Path to save the plot.
        """
        importance_scores = self.get_feature_importance(class_idx)
        
        # Sort by importance
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        features, scores = zip(*sorted_features)
        
        # Convert scores to simple floats (average across classes if needed)
        scores = [float(np.mean(score)) if isinstance(score, (list, np.ndarray)) else float(score) for score in scores]
        
        plt.figure(figsize=self.config.visualization.figure_size)
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Mean |SHAP value|')
        plt.title(f'Feature Importance (Class {class_idx if class_idx is not None else "All"})')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.visualization.dpi, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")
        
        plt.show()
    
    def save_explanations(self, filepath: str) -> None:
        """Save SHAP explanations to file.
        
        Args:
            filepath: Path to save explanations.
        """
        explanations = {
            "shap_values": self.shap_values.tolist() if isinstance(self.shap_values, np.ndarray) else self.shap_values,
            "feature_names": self.feature_names,
            "config": {
                "explainer_type": self.config.shap.explainer_type,
                "background_samples": self.config.shap.background_samples,
                "max_samples": self.config.shap.max_samples
            }
        }
        
        np.save(filepath, explanations)
        logger.info(f"Saved explanations to {filepath}")
    
    def load_explanations(self, filepath: str) -> None:
        """Load SHAP explanations from file.
        
        Args:
            filepath: Path to load explanations from.
        """
        explanations = np.load(filepath, allow_pickle=True).item()
        
        self.shap_values = explanations["shap_values"]
        self.feature_names = explanations["feature_names"]
        
        logger.info(f"Loaded explanations from {filepath}")
    
    def get_explanation_summary(self) -> Dict[str, Any]:
        """Get summary of explanations.
        
        Returns:
            Dictionary with explanation summary.
        """
        if self.shap_values is None:
            return {"status": "No explanations computed"}
        
        summary = {
            "explainer_type": self.config.shap.explainer_type,
            "n_samples": len(self.shap_values) if isinstance(self.shap_values, np.ndarray) else len(self.shap_values[0]),
            "n_features": len(self.feature_names) if self.feature_names else "Unknown",
            "feature_names": self.feature_names,
            "is_multiclass": isinstance(self.shap_values, list)
        }
        
        if isinstance(self.shap_values, list):
            summary["n_classes"] = len(self.shap_values)
        
        return summary
