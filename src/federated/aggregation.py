"""
Aggregation methods for federated learning
"""

import torch
from typing import Dict, List, Optional
import numpy as np


def federated_averaging(
    client_parameters: List[Dict[str, torch.Tensor]],
    client_weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    FedAvg: Average model parameters from multiple clients
    
    Args:
        client_parameters: List of parameter dictionaries from clients
        client_weights: Optional weights for each client (e.g., based on dataset size)
    
    Returns:
        Aggregated parameters
    """
    if not client_parameters:
        raise ValueError("No client parameters to aggregate")
    
    # Equal weights if not provided
    if client_weights is None:
        client_weights = [1.0 / len(client_parameters)] * len(client_parameters)
    else:
        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
    
    # Initialize aggregated parameters
    aggregated = {}
    
    # Get parameter names from first client
    param_names = client_parameters[0].keys()
    
    # Average each parameter
    for name in param_names:
        # Stack parameters from all clients
        params = torch.stack([
            client_params[name] for client_params in client_parameters
        ])
        
        # Weighted average
        weights_tensor = torch.tensor(
            client_weights, dtype=params.dtype, device=params.device
        ).view(-1, *([1] * (params.dim() - 1)))
        
        aggregated[name] = (params * weights_tensor).sum(dim=0)
    
    return aggregated


def weighted_averaging(
    client_parameters: List[Dict[str, torch.Tensor]],
    client_samples: List[int],
) -> Dict[str, torch.Tensor]:
    """
    Weighted averaging based on number of samples at each client
    
    Args:
        client_parameters: List of parameter dictionaries
        client_samples: Number of samples at each client
    
    Returns:
        Aggregated parameters
    """
    # Calculate weights based on sample counts
    total_samples = sum(client_samples)
    weights = [n / total_samples for n in client_samples]
    
    return federated_averaging(client_parameters, weights)


def fedprox_aggregation(
    client_parameters: List[Dict[str, torch.Tensor]],
    server_parameters: Dict[str, torch.Tensor],
    client_weights: Optional[List[float]] = None,
    mu: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """
    FedProx aggregation with proximal term
    
    Args:
        client_parameters: Client model parameters
        server_parameters: Current server parameters
        client_weights: Client weights
        mu: Proximal term coefficient
    
    Returns:
        Aggregated parameters
    """
    # Standard FedAvg
    aggregated = federated_averaging(client_parameters, client_weights)
    
    # Add proximal term (pull towards server model)
    for name in aggregated.keys():
        aggregated[name] = (
            aggregated[name] + mu * server_parameters[name]
        ) / (1 + mu)
    
    return aggregated


def secure_aggregation(
    client_parameters: List[Dict[str, torch.Tensor]],
    client_weights: Optional[List[float]] = None,
    noise_scale: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Secure aggregation with optional noise for additional privacy
    
    Args:
        client_parameters: Client parameters
        client_weights: Client weights
        noise_scale: Scale of Gaussian noise to add
    
    Returns:
        Aggregated parameters with noise
    """
    # Standard aggregation
    aggregated = federated_averaging(client_parameters, client_weights)
    
    # Add noise if specified
    if noise_scale > 0:
        for name, param in aggregated.items():
            noise = torch.randn_like(param) * noise_scale
            aggregated[name] = param + noise
    
    return aggregated


class AdaptiveAggregator:
    """
    Adaptive aggregation with momentum and learning rate scheduling
    """
    
    def __init__(
        self,
        momentum: float = 0.9,
        learning_rate: float = 1.0,
    ):
        """
        Initialize adaptive aggregator
        
        Args:
            momentum: Momentum coefficient
            learning_rate: Server learning rate
        """
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.velocity = None
    
    def aggregate(
        self,
        client_parameters: List[Dict[str, torch.Tensor]],
        server_parameters: Dict[str, torch.Tensor],
        client_weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate with momentum
        
        Args:
            client_parameters: Client parameters
            server_parameters: Current server parameters
            client_weights: Client weights
        
        Returns:
            Updated server parameters
        """
        # Compute standard aggregation
        aggregated = federated_averaging(client_parameters, client_weights)
        
        # Initialize velocity if needed
        if self.velocity is None:
            self.velocity = {
                name: torch.zeros_like(param)
                for name, param in aggregated.items()
            }
        
        # Update with momentum
        updated = {}
        for name in aggregated.keys():
            # Compute pseudo-gradient
            pseudo_grad = server_parameters[name] - aggregated[name]
            
            # Update velocity
            self.velocity[name] = (
                self.momentum * self.velocity[name] + pseudo_grad
            )
            
            # Update parameters
            updated[name] = (
                server_parameters[name] -
                self.learning_rate * self.velocity[name]
            )
        
        return updated
