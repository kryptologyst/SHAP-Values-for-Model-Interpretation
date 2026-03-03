"""Simple neural network implementation for XAI experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SimpleNeuralNetwork(nn.Module):
    """Simple feedforward neural network for classification."""
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [64, 32],
        output_size: int = 3,
        dropout_rate: float = 0.2,
        activation: str = "relu"
    ):
        """Initialize neural network.
        
        Args:
            input_size: Number of input features.
            hidden_sizes: List of hidden layer sizes.
            output_size: Number of output classes.
            dropout_rate: Dropout rate for regularization.
            activation: Activation function ("relu", "tanh", "sigmoid").
        """
        super(SimpleNeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                self._get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function module.
        
        Args:
            activation: Activation function name.
            
        Returns:
            Activation function module.
        """
        if activation == "relu":
            return nn.ReLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output logits.
        """
        return self.network(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities.
        
        Args:
            x: Input tensor.
            
        Returns:
            Prediction probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predictions.
        
        Args:
            x: Input tensor.
            
        Returns:
            Predicted class labels.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
