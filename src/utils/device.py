"""Utility functions for device management and deterministic seeding."""

import random
import numpy as np
import torch
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


def get_device(device: str = "auto") -> torch.device:
    """Get the best available device for computation.
    
    Args:
        device: Device preference ("auto", "cpu", "cuda", "mps").
        
    Returns:
        torch.device object for the selected device.
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("Using CUDA device")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using MPS device (Apple Silicon)")
        else:
            device = "cpu"
            logger.info("Using CPU device")
    
    return torch.device(device)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def get_memory_info() -> dict:
    """Get memory information for the current device.
    
    Returns:
        Dictionary with memory information.
    """
    info = {}
    
    if torch.cuda.is_available():
        info["cuda_available"] = True
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_current_device"] = torch.cuda.current_device()
        info["cuda_device_name"] = torch.cuda.get_device_name()
        info["cuda_memory_allocated"] = torch.cuda.memory_allocated()
        info["cuda_memory_reserved"] = torch.cuda.memory_reserved()
        info["cuda_max_memory_allocated"] = torch.cuda.max_memory_allocated()
    else:
        info["cuda_available"] = False
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info["mps_available"] = True
    else:
        info["mps_available"] = False
    
    return info


def clear_memory() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA memory cache cleared")
