"""
A2C Agent (Advantage Actor-Critic)
Implements Fed-A2C with support for differential privacy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional

from .networks import ActorCriticNetwork, create_network


class A2CAgent:
    """
    A2C agent for federated learning with differential privacy
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[Dict] = None,
        device: str = "cpu",
    ):
        """
        Initialize A2C agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary
            device: Device to run on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        
        # Default configuration
        if config is None:
            config = {}
        
        self.config = {
            "learning_rate": 7e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
            "n_steps": 5,
            **config
        }
        
        # Create network
        network_config = config.get("network", {})
        self.network = create_network(
            "actor_critic",
            state_dim,
            action_dim,
            network_config
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config["learning_rate"]
        )
        
        # N-step buffer
        self.buffer = A2CBuffer(
            n_steps=self.config["n_steps"],
            gamma=self.config["gamma"],
            gae_lambda=self.config["gae_lambda"],
        )
        
        # Training statistics
        self.training_stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "total_loss": [],
            "grad_norm": [],
        }
    
    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, Dict]:
        """Select action given state"""
        self.network.eval()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, entropy, value = self.network.get_action_and_value(
                state_tensor
            )
            
            if deterministic:
                logits, _, _ = self.network(state_tensor)
                action = torch.argmax(logits, dim=-1)
        
        info = {
            "log_prob": log_prob.cpu().item(),
            "value": value.cpu().item(),
            "entropy": entropy.cpu().item(),
        }
        
        return action.cpu().item(), info
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Dict,
    ):
        """Store transition in buffer"""
        self.buffer.add(
            state=state,
            action=action,
            reward=reward,
            value=info["value"],
            log_prob=info["log_prob"],
            done=done,
        )
    
    def update(self, next_state: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Update policy using A2C algorithm
        
        Args:
            next_state: Next state for bootstrap (if not done)
        
        Returns:
            Training statistics
        """
        if not self.buffer.is_ready():
            return {}
        
        self.network.train()
        
        # Get bootstrap value
        bootstrap_value = 0.0
        if next_state is not None:
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                bootstrap_value = self.network.get_value(next_state_tensor).item()
        
        # Get all data from buffer
        states, actions, old_log_probs, returns, advantages = self.buffer.get(
            bootstrap_value
        )
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass
        _, new_log_probs, entropy, values = self.network.get_action_and_value(
            states, actions
        )
        
        # Policy loss
        policy_loss = -(new_log_probs * advantages).mean()
        
        # Value loss
        value_loss = nn.functional.mse_loss(values, returns)
        
        # Entropy bonus
        entropy_loss = -entropy.mean()
        
        # Total loss
        loss = (
            policy_loss +
            self.config["value_coef"] * value_loss +
            self.config["entropy_coef"] * entropy_loss
        )
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(
            self.network.parameters(),
            self.config["max_grad_norm"]
        )
        
        self.optimizer.step()
        
        # Record statistics
        stats = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item(),
            "total_loss": loss.item(),
            "grad_norm": grad_norm.item(),
        }
        
        # Update training statistics
        for key, val in stats.items():
            self.training_stats[key].append(val)
        
        # Clear buffer
        self.buffer.reset()
        
        return stats
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get network parameters (for federated learning)"""
        return {
            name: param.clone().detach()
            for name, param in self.network.state_dict().items()
        }
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set network parameters (for federated learning)"""
        self.network.load_state_dict(parameters)
    
    def save(self, path: str):
        """Save agent to file"""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
            "training_stats": self.training_stats,
        }, path)
    
    def load(self, path: str):
        """Load agent from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.config.update(checkpoint.get("config", {}))
        self.training_stats = checkpoint.get("training_stats", self.training_stats)


class A2CBuffer:
    """N-step buffer for A2C"""
    
    def __init__(
        self,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """Initialize buffer"""
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()
    
    def reset(self):
        """Reset buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """Add transition"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def is_ready(self) -> bool:
        """Check if buffer has enough steps"""
        return len(self.states) >= self.n_steps
    
    def get(self, bootstrap_value: float) -> Tuple[np.ndarray, ...]:
        """
        Get all data and compute returns and advantages
        
        Args:
            bootstrap_value: Value estimate for next state
        
        Returns:
            states, actions, log_probs, returns, advantages
        """
        n = len(self.states)
        
        # Compute GAE advantages
        advantages = np.zeros(n, dtype=np.float32)
        last_gae_lam = 0
        
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = bootstrap_value
                next_non_terminal = 1 - int(self.dones[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1 - int(self.dones[t])
            
            delta = (
                self.rewards[t] +
                self.gamma * next_value * next_non_terminal -
                self.values[t]
            )
            advantages[t] = last_gae_lam = (
                delta +
                self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
        
        # Compute returns
        returns = advantages + np.array(self.values)
        
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.log_probs),
            returns,
            advantages,
        )
