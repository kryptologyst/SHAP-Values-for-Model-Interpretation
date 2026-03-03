"""Main pipeline for SHAP XAI project."""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.device import set_seed, get_device
from src.data.loader import DataLoader
from src.models.manager import ModelManager
from src.explainers.shap_explainer import SHAPExplainer
from src.metrics.xai_metrics import XAIMetrics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main pipeline execution."""
    logger.info("Starting SHAP XAI Pipeline")
    
    # Load configuration
    config = load_config()
    logger.info(f"Loaded configuration: {config.data.dataset_name}")
    
    # Set random seed for reproducibility
    set_seed(config.data.random_state)
    
    # Initialize components
    data_loader = DataLoader(config)
    model_manager = ModelManager(config)
    shap_explainer = SHAPExplainer(config)
    xai_metrics = XAIMetrics(config)
    
    # Load and preprocess data
    logger.info("Loading dataset...")
    X, y, feature_names, target_names = data_loader.load_dataset()
    
    # Preprocess data
    X_processed, y_processed = data_loader.preprocess_data(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test = data_loader.split_data(X_processed, y_processed)
    
    # Update feature names in data loader
    data_loader.feature_names = feature_names
    data_loader.target_names = target_names
    
    # Save metadata
    metadata_path = config.data_dir / "metadata.json"
    data_loader.save_metadata(str(metadata_path))
    
    # Train model
    logger.info("Training model...")
    model = model_manager.train_model(X_train, y_train)
    
    # Evaluate model
    logger.info("Evaluating model...")
    model_metrics = model_manager.evaluate_model(X_test, y_test)
    logger.info(f"Model accuracy: {model_metrics['accuracy']:.4f}")
    
    # Create SHAP explainer
    logger.info("Creating SHAP explainer...")
    explainer = shap_explainer.create_explainer(model, X_train, feature_names)
    
    # Compute SHAP values
    logger.info("Computing SHAP values...")
    shap_values = shap_explainer.compute_shap_values(X_test)
    
    # Generate explanations
    logger.info("Generating explanations...")
    
    # Summary plot
    summary_path = config.assets_dir / "plots" / "shap_summary.png"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    shap_explainer.plot_summary(save_path=str(summary_path))
    
    # Feature importance plot
    importance_path = config.assets_dir / "plots" / "feature_importance.png"
    shap_explainer.plot_feature_importance(save_path=str(importance_path))
    
    # Waterfall plot for first instance
    waterfall_path = config.assets_dir / "plots" / "shap_waterfall.png"
    shap_explainer.plot_waterfall(0, save_path=str(waterfall_path))
    
    # Get feature importance
    feature_importance = shap_explainer.get_feature_importance()
    logger.info("Feature importance:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        # Handle multi-class importance (convert to float if it's a list)
        if isinstance(importance, (list, np.ndarray)):
            importance = float(np.mean(importance))
        logger.info(f"  {feature}: {importance:.4f}")
    
    # Explain a single prediction
    instance_explanation = shap_explainer.explain_prediction(X_test[0])
    logger.info(f"Explanation for first test instance: {instance_explanation}")
    
    # Evaluate explanations
    logger.info("Evaluating explanations...")
    evaluation_metrics = xai_metrics.compute_comprehensive_metrics(
        model, X_test, y_test, shap_values, feature_names
    )
    
    # Log evaluation results
    logger.info("Evaluation Results:")
    if "faithfulness_deletion" in evaluation_metrics:
        deletion_scores = evaluation_metrics["faithfulness_deletion"]["deletion_scores"]
        logger.info(f"  Deletion test scores: {deletion_scores}")
    
    if "faithfulness_insertion" in evaluation_metrics:
        insertion_scores = evaluation_metrics["faithfulness_insertion"]["insertion_scores"]
        logger.info(f"  Insertion test scores: {insertion_scores}")
    
    # Save explanations
    explanations_path = config.assets_dir / "explanations" / "shap_values.npy"
    explanations_path.parent.mkdir(parents=True, exist_ok=True)
    shap_explainer.save_explanations(str(explanations_path))
    
    # Save model
    model_path = config.checkpoints_dir / f"{config.model.model_type}_model.pkl"
    model_manager.save_model(str(model_path))
    
    # Create results summary
    results_summary = {
        "model_accuracy": model_metrics["accuracy"],
        "feature_importance": feature_importance,
        "evaluation_metrics": evaluation_metrics,
        "explanation_summary": shap_explainer.get_explanation_summary()
    }
    
    # Save results summary
    results_path = config.assets_dir / "results_summary.json"
    import json
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    logger.info("Pipeline completed successfully!")
    logger.info(f"Results saved to: {config.assets_dir}")
    
    return results_summary


if __name__ == "__main__":
    results = main()
