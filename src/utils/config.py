"""Configuration management for SHAP XAI project."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
from omegaconf import OmegaConf


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    dataset_name: str = "iris"
    test_size: float = 0.3
    random_state: int = 42
    feature_scaling: bool = True
    categorical_encoding: str = "onehot"  # onehot, ordinal, target
    imputation_strategy: str = "median"  # mean, median, mode, constant
    sensitive_attributes: List[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    """Configuration for model training."""
    model_type: str = "random_forest"  # random_forest, xgboost, neural_network
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    random_state: int = 42
    device: str = "auto"  # auto, cpu, cuda, mps


@dataclass
class SHAPConfig:
    """Configuration for SHAP explanations."""
    explainer_type: str = "tree"  # tree, kernel, deep, linear
    background_samples: int = 100
    max_samples: int = 1000
    feature_perturbation: str = "tree_path_dependent"
    model_output: str = "raw"  # raw, probability, logit
    link: str = "identity"  # identity, logit, log


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    faithfulness_tests: bool = True
    stability_tests: bool = True
    fidelity_tests: bool = True
    n_bootstrap_samples: int = 100
    deletion_percentages: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.5])
    insertion_percentages: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.5])


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    save_plots: bool = True
    plot_format: str = "png"  # png, pdf, svg
    dpi: int = 300
    style: str = "seaborn-v0_8"
    color_palette: str = "viridis"
    figure_size: tuple = (10, 6)


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    shap: SHAPConfig = field(default_factory=SHAPConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # Paths
    data_dir: str = "data"
    assets_dir: str = "assets"
    logs_dir: str = "logs"
    checkpoints_dir: str = "checkpoints"
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Convert string paths to Path objects
        self.data_dir = Path(self.data_dir)
        self.assets_dir = Path(self.assets_dir)
        self.logs_dir = Path(self.logs_dir)
        self.checkpoints_dir = Path(self.checkpoints_dir)
        
        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.assets_dir, self.logs_dir, self.checkpoints_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file or return default config.
    
    Args:
        config_path: Path to YAML configuration file.
        
    Returns:
        Config object with loaded settings.
    """
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return OmegaConf.structured(Config(**config_dict))
    else:
        return Config()


def save_config(config: Config, config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save.
        config_path: Path where to save the configuration.
    """
    config_dict = OmegaConf.structured(config)
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
