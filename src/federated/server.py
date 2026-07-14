"""
Federated Server
Coordinates federated learning across multiple clients
"""

import torch
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

from .aggregation import federated_averaging, weighted_averaging, AdaptiveAggregator
from ..privacy.privacy_accountant import PrivacyAccountant


class FederatedServer:
    """
    Federated learning server
    Manages global model and coordinates client training
    """
    
    def __init__(
        self,
        global_model: torch.nn.Module,
        aggregation_method: str = "fedavg",
        privacy_accountant: Optional[PrivacyAccountant] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize federated server
        
        Args:
            global_model: Global model to be trained
            aggregation_method: Aggregation strategy ('fedavg', 'weighted', 'adaptive')
            privacy_accountant: Privacy tracking
            config: Configuration dictionary
        """
        self.global_model = global_model
        self.aggregation_method = aggregation_method
        self.privacy_accountant = privacy_accountant
        
        if config is None:
            config = {}
        self.config = config
        
        # Initialize aggregator
        if aggregation_method == "adaptive":
            self.aggregator = AdaptiveAggregator(
                momentum=config.get("momentum", 0.9),
                learning_rate=config.get("server_lr", 1.0),
            )
        else:
            self.aggregator = None
        
        # Training state
        self.current_round = 0
        self.client_models = {}
        self.client_weights = {}
        self.training_history = {
            "rounds": [],
            "avg_rewards": [],
            "avg_losses": [],
            "privacy_epsilon": [],
        }
        
        self.logger = logging.getLogger("FederatedServer")
    
    def get_global_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current global model parameters"""
        return {
            name: param.clone().detach()
            for name, param in self.global_model.state_dict().items()
        }
    
    def aggregate_client_updates(
        self,
        client_parameters: Dict[str, Dict[str, torch.Tensor]],
        client_sample_counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate updates from multiple clients
        
        Args:
            client_parameters: Dict mapping client_id to parameters
            client_sample_counts: Optional dict of sample counts per client
        
        Returns:
            Aggregated parameters
        """
        client_params_list = list(client_parameters.values())
        client_ids = list(client_parameters.keys())
        
        if self.aggregation_method == "fedavg":
            aggregated = federated_averaging(client_params_list)
        
        elif self.aggregation_method == "weighted":
            if client_sample_counts is None:
                raise ValueError("Sample counts required for weighted aggregation")
            
            sample_counts = [
                client_sample_counts[cid] for cid in client_ids
            ]
            aggregated = weighted_averaging(client_params_list, sample_counts)
        
        elif self.aggregation_method == "adaptive":
            if self.aggregator is None:
                raise ValueError("Adaptive aggregator not initialized")
            
            current_params = self.get_global_parameters()
            aggregated = self.aggregator.aggregate(
                client_params_list,
                current_params,
                None,  # Can add client weights here
            )
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        return aggregated
    
    def update_global_model(
        self,
        client_parameters: Dict[str, Dict[str, torch.Tensor]],
        client_sample_counts: Optional[Dict[str, int]] = None,
    ):
        """
        Update global model with aggregated client parameters
        
        Args:
            client_parameters: Client parameter dictionaries
            client_sample_counts: Optional sample counts
        """
        # Aggregate client updates
        aggregated_params = self.aggregate_client_updates(
            client_parameters, client_sample_counts
        )
        
        # Update global model
        self.global_model.load_state_dict(aggregated_params)
        
        # Update round counter
        self.current_round += 1
        
        self.logger.info(f"Global model updated - Round {self.current_round}")
    
    def select_clients(
        self,
        all_clients: List[str],
        sampling_rate: float = 1.0,
        min_clients: int = 1,
    ) -> List[str]:
        """
        Select clients for current round
        
        Args:
            all_clients: List of all available client IDs
            sampling_rate: Fraction of clients to select
            min_clients: Minimum number of clients
        
        Returns:
            List of selected client IDs
        """
        num_clients = max(min_clients, int(len(all_clients) * sampling_rate))
        
        # Random sampling
        import random
        selected = random.sample(all_clients, num_clients)
        
        self.logger.info(f"Selected {len(selected)}/{len(all_clients)} clients")
        
        return selected
    
    def track_privacy(
        self,
        client_id: str,
        noise_multiplier: float,
        sampling_rate: float,
        steps: int,
    ):
        """Track privacy for a client"""
        if self.privacy_accountant is not None:
            self.privacy_accountant.track_privacy(
                client_id, noise_multiplier, sampling_rate, steps
            )
    
    def get_privacy_status(self) -> Dict[str, Any]:
        """Get current privacy status"""
        if self.privacy_accountant is None:
            return {"enabled": False}
        
        return {
            "enabled": True,
            **self.privacy_accountant.get_privacy_report()
        }
    
    def save_checkpoint(self, path: str):
        """Save server state"""
        checkpoint = {
            "round": self.current_round,
            "model_state": self.global_model.state_dict(),
            "config": self.config,
            "history": self.training_history,
        }
        
        if self.privacy_accountant is not None:
            checkpoint["privacy_report"] = self.privacy_accountant.get_privacy_report()
        
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load server state"""
        checkpoint = torch.load(path)
        
        self.current_round = checkpoint["round"]
        self.global_model.load_state_dict(checkpoint["model_state"])
        self.config.update(checkpoint.get("config", {}))
        self.training_history = checkpoint.get("history", self.training_history)
        
        self.logger.info(f"Checkpoint loaded from {path}")
