"""Evaluation metrics for XAI explanations."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error
from scipy.stats import spearmanr, kendalltau
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class XAIMetrics:
    """Comprehensive evaluation metrics for XAI explanations."""
    
    def __init__(self, config):
        """Initialize XAI metrics with configuration.
        
        Args:
            config: Configuration object with evaluation settings.
        """
        self.config = config
    
    def faithfulness_deletion(self, model: Any, X: np.ndarray, y: np.ndarray, 
                             shap_values: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Compute faithfulness using deletion test.
        
        Args:
            model: Trained model.
            X: Test features.
            y: Test targets.
            shap_values: SHAP values for explanations.
            feature_names: Names of features.
            
        Returns:
            Dictionary with deletion test metrics.
        """
        original_predictions = model.predict_proba(X)
        original_accuracy = np.mean(np.argmax(original_predictions, axis=1) == y)
        
        deletion_scores = []
        
        for percentage in self.config.evaluation.deletion_percentages:
            # Get most important features to delete
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            n_features_to_delete = int(len(feature_names) * percentage)
            features_to_delete = np.argsort(feature_importance)[-n_features_to_delete:]
            
            # Create modified dataset
            X_modified = X.copy()
            X_modified[:, features_to_delete] = 0  # Set to zero
            
            # Compute new predictions
            modified_predictions = model.predict_proba(X_modified)
            modified_accuracy = np.mean(np.argmax(modified_predictions, axis=1) == y)
            
            deletion_score = original_accuracy - modified_accuracy
            deletion_scores.append(deletion_score)
        
        return {
            "deletion_scores": deletion_scores,
            "deletion_percentages": self.config.evaluation.deletion_percentages,
            "original_accuracy": original_accuracy
        }
    
    def faithfulness_insertion(self, model: Any, X: np.ndarray, y: np.ndarray,
                              shap_values: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Compute faithfulness using insertion test.
        
        Args:
            model: Trained model.
            X: Test features.
            y: Test targets.
            shap_values: SHAP values for explanations.
            feature_names: Names of features.
            
        Returns:
            Dictionary with insertion test metrics.
        """
        # Baseline: predict with no features
        baseline_predictions = np.full((len(X), len(np.unique(y))), 1.0 / len(np.unique(y)))
        baseline_accuracy = np.mean(np.argmax(baseline_predictions, axis=1) == y)
        
        insertion_scores = []
        
        for percentage in self.config.evaluation.insertion_percentages:
            # Get most important features to insert
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            n_features_to_insert = int(len(feature_names) * percentage)
            features_to_insert = np.argsort(feature_importance)[-n_features_to_insert:]
            
            # Create modified dataset with only important features
            X_modified = np.zeros_like(X)
            X_modified[:, features_to_insert] = X[:, features_to_insert]
            
            # Compute predictions
            modified_predictions = model.predict_proba(X_modified)
            modified_accuracy = np.mean(np.argmax(modified_predictions, axis=1) == y)
            
            insertion_score = modified_accuracy - baseline_accuracy
            insertion_scores.append(insertion_score)
        
        return {
            "insertion_scores": insertion_scores,
            "insertion_percentages": self.config.evaluation.insertion_percentages,
            "baseline_accuracy": baseline_accuracy
        }
    
    def stability_across_seeds(self, shap_values_list: List[np.ndarray]) -> Dict[str, float]:
        """Compute stability of explanations across different random seeds.
        
        Args:
            shap_values_list: List of SHAP values computed with different seeds.
            
        Returns:
            Dictionary with stability metrics.
        """
        if len(shap_values_list) < 2:
            return {"error": "Need at least 2 sets of SHAP values"}
        
        # Compute pairwise correlations
        correlations = []
        for i in range(len(shap_values_list)):
            for j in range(i + 1, len(shap_values_list)):
                # Flatten arrays for correlation
                vals1 = shap_values_list[i].flatten()
                vals2 = shap_values_list[j].flatten()
                
                # Spearman correlation
                corr, _ = spearmanr(vals1, vals2)
                correlations.append(corr)
        
        # Compute Kendall's tau
        tau_values = []
        for i in range(len(shap_values_list)):
            for j in range(i + 1, len(shap_values_list)):
                vals1 = shap_values_list[i].flatten()
                vals2 = shap_values_list[j].flatten()
                
                tau, _ = kendalltau(vals1, vals2)
                tau_values.append(tau)
        
        return {
            "mean_spearman_correlation": np.mean(correlations),
            "std_spearman_correlation": np.std(correlations),
            "mean_kendall_tau": np.mean(tau_values),
            "std_kendall_tau": np.std(tau_values),
            "n_comparisons": len(correlations)
        }
    
    def feature_importance_stability(self, shap_values_list: List[np.ndarray], 
                                   feature_names: List[str]) -> Dict[str, Any]:
        """Compute stability of feature importance rankings.
        
        Args:
            shap_values_list: List of SHAP values computed with different seeds.
            feature_names: Names of features.
            
        Returns:
            Dictionary with feature importance stability metrics.
        """
        rankings = []
        
        for shap_values in shap_values_list:
            # Compute feature importance
            importance = np.mean(np.abs(shap_values), axis=0)
            ranking = np.argsort(importance)[::-1]  # Descending order
            rankings.append(ranking)
        
        # Compute ranking stability
        ranking_correlations = []
        for i in range(len(rankings)):
            for j in range(i + 1, len(rankings)):
                corr, _ = spearmanr(rankings[i], rankings[j])
                ranking_correlations.append(corr)
        
        # Top-k stability
        top_k_stability = {}
        for k in [3, 5, 10]:
            if k > len(feature_names):
                continue
            
            top_k_overlaps = []
            for i in range(len(rankings)):
                for j in range(i + 1, len(rankings)):
                    top_k_i = set(rankings[i][:k])
                    top_k_j = set(rankings[j][:k])
                    overlap = len(top_k_i.intersection(top_k_j)) / k
                    top_k_overlaps.append(overlap)
            
            top_k_stability[f"top_{k}_overlap"] = np.mean(top_k_overlaps)
        
        return {
            "mean_ranking_correlation": np.mean(ranking_correlations),
            "std_ranking_correlation": np.std(ranking_correlations),
            "top_k_stability": top_k_stability
        }
    
    def surrogate_fidelity(self, original_model: Any, surrogate_model: Any, 
                          X_test: np.ndarray) -> Dict[str, float]:
        """Compute fidelity of surrogate model to original model.
        
        Args:
            original_model: Original model to approximate.
            surrogate_model: Surrogate model (e.g., decision tree).
            X_test: Test data.
            
        Returns:
            Dictionary with fidelity metrics.
        """
        # Get predictions from both models
        original_preds = original_model.predict_proba(X_test)
        surrogate_preds = surrogate_model.predict_proba(X_test)
        
        # Compute fidelity metrics
        mse = mean_squared_error(original_preds.flatten(), surrogate_preds.flatten())
        
        # Agreement on predictions
        original_labels = np.argmax(original_preds, axis=1)
        surrogate_labels = np.argmax(surrogate_preds, axis=1)
        agreement = np.mean(original_labels == surrogate_labels)
        
        return {
            "mse": mse,
            "agreement": agreement,
            "rmse": np.sqrt(mse)
        }
    
    def compute_comprehensive_metrics(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                                     shap_values: np.ndarray, feature_names: List[str],
                                     shap_values_list: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics.
        
        Args:
            model: Trained model.
            X_test: Test features.
            y_test: Test targets.
            shap_values: SHAP values for explanations.
            feature_names: Names of features.
            shap_values_list: List of SHAP values for stability testing.
            
        Returns:
            Dictionary with all evaluation metrics.
        """
        metrics = {}
        
        # Faithfulness tests
        if self.config.evaluation.faithfulness_tests:
            metrics["faithfulness_deletion"] = self.faithfulness_deletion(
                model, X_test, y_test, shap_values, feature_names
            )
            metrics["faithfulness_insertion"] = self.faithfulness_insertion(
                model, X_test, y_test, shap_values, feature_names
            )
        
        # Stability tests
        if self.config.evaluation.stability_tests and shap_values_list is not None:
            metrics["stability_across_seeds"] = self.stability_across_seeds(shap_values_list)
            metrics["feature_importance_stability"] = self.feature_importance_stability(
                shap_values_list, feature_names
            )
        
        # Basic explanation statistics
        metrics["explanation_stats"] = {
            "mean_abs_shap": np.mean(np.abs(shap_values)),
            "std_abs_shap": np.std(np.abs(shap_values)),
            "max_abs_shap": np.max(np.abs(shap_values)),
            "min_abs_shap": np.min(np.abs(shap_values))
        }
        
        return metrics
    
    def create_leaderboard(self, metrics_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Create a leaderboard from multiple model metrics.
        
        Args:
            metrics_dict: Dictionary mapping model names to their metrics.
            
        Returns:
            DataFrame with leaderboard.
        """
        leaderboard_data = []
        
        for model_name, metrics in metrics_dict.items():
            row = {"model": model_name}
            
            # Extract key metrics
            if "faithfulness_deletion" in metrics:
                row["deletion_score"] = np.mean(metrics["faithfulness_deletion"]["deletion_scores"])
            
            if "faithfulness_insertion" in metrics:
                row["insertion_score"] = np.mean(metrics["faithfulness_insertion"]["insertion_scores"])
            
            if "stability_across_seeds" in metrics:
                row["stability_correlation"] = metrics["stability_across_seeds"]["mean_spearman_correlation"]
            
            if "explanation_stats" in metrics:
                row["mean_abs_shap"] = metrics["explanation_stats"]["mean_abs_shap"]
            
            leaderboard_data.append(row)
        
        return pd.DataFrame(leaderboard_data)
