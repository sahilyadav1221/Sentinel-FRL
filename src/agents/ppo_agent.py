"""
PPO Agent (Proximal Policy Optimization)
Implements Fed-PPO with support for differential privacy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque

from .networks import ActorCriticNetwork, create_network


class PPOAgent:
    """
    PPO agent for federated learning with differential privacy
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[Dict] = None,
        device: str = "cpu",
    ):
        """
        Initialize PPO agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary
            device: Device to run on ('cpu', 'cuda', 'mps')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        
        # Default configuration
        if config is None:
            config = {}
        
        self.config = {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
            "n_epochs": 10,
            "batch_size": 64,
            "buffer_size": 2048,
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
        
        # Experience buffer
        self.buffer = PPOBuffer(
            buffer_size=self.config["buffer_size"],
            state_dim=state_dim,
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
        """
        Select action given state
        
        Args:
            state: Current state
            deterministic: Whether to select action deterministically
        
        Returns:
            action: Selected action
            info: Additional information (log_prob, value, etc.)
        """
        self.network.eval()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, entropy, value = self.network.get_action_and_value(
                state_tensor
            )
            
            if deterministic:
                # Use greedy action
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
    
    def update(self) -> Dict[str, float]:
        """
        Update policy using PPO algorithm
        
        Returns:
            Training statistics
        """
        if not self.buffer.is_full():
            return {}
        
        self.network.train()
        
        # Get all data from buffer
        states, actions, old_log_probs, returns, advantages = self.buffer.get()
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        epoch_stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "total_loss": [],
            "grad_norm": [],
        }
        
        batch_size = self.config["batch_size"]
        n_samples = len(states)
        indices = np.arange(n_samples)
        
        for epoch in range(self.config["n_epochs"]):
            np.random.shuffle(indices)
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Forward pass
                _, new_log_probs, entropy, values = self.network.get_action_and_value(
                    batch_states, batch_actions
                )
                
                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config["clip_range"],
                    1 + self.config["clip_range"]
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, batch_returns)
                
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
                epoch_stats["policy_loss"].append(policy_loss.item())
                epoch_stats["value_loss"].append(value_loss.item())
                epoch_stats["entropy"].append(-entropy_loss.item())
                epoch_stats["total_loss"].append(loss.item())
                epoch_stats["grad_norm"].append(grad_norm.item())
        
        # Average statistics
        stats = {key: np.mean(val) for key, val in epoch_stats.items()}
        
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


class PPOBuffer:
    """
    Experience buffer for PPO with GAE
    """
    
    def __init__(
        self,
        buffer_size: int,
        state_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """Initialize buffer"""
        self.buffer_size = buffer_size
        self.state_dim = state_dim
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
        self.ptr = 0
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """Add transition to buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.ptr += 1
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return self.ptr >= self.buffer_size
    
    def get(self) -> Tuple[np.ndarray, ...]:
        """
        Get all data and compute returns and advantages
        
        Returns:
            states, actions, log_probs, returns, advantages
        """
        # Compute GAE advantages
        advantages = np.zeros(self.ptr, dtype=np.float32)
        last_gae_lam = 0
        
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = 0
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
        returns = advantages + np.array(self.values[:self.ptr])
        
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.log_probs),
            returns,
            advantages,
        )
