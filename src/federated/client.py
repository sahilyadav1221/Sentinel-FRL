"""
Federated Client
Represents a single edge gateway/site in federated learning
"""

import torch
import numpy as np
from typing import Dict, Optional, Any, List
import logging
from pathlib import Path

from ..environment.gateway_env import GatewayAnomalyEnv
from ..agents.ppo_agent import PPOAgent
from ..agents.a2c_agent import A2CAgent
from ..privacy.dp_sgd import make_private


class FederatedClient:
    """
    Federated learning client
    Trains local model on site-specific data
    """
    
    def __init__(
        self,
        client_id: str,
        env: GatewayAnomalyEnv,
        agent_type: str = "ppo",
        agent_config: Optional[Dict] = None,
        dp_config: Optional[Dict] = None,
        device: str = "cpu",
    ):
        """
        Initialize federated client
        
        Args:
            client_id: Unique identifier for this client
            env: Environment for this client
            agent_type: Type of agent ('ppo' or 'a2c')
            agent_config: Agent configuration
            dp_config: Differential privacy configuration
            device: Device to run on
        """
        self.client_id = client_id
        self.env = env
        self.agent_type = agent_type
        self.device = device
        
        # Initialize logger first
        self.logger = logging.getLogger(f"Client-{client_id}")
        
        # Initialize agent
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        if agent_type.lower() == "ppo":
            self.agent = PPOAgent(state_dim, action_dim, agent_config, device)
        elif agent_type.lower() == "a2c":
            self.agent = A2CAgent(state_dim, action_dim, agent_config, device)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Apply differential privacy if configured
        self.dp_config = dp_config
        if dp_config is not None and dp_config.get("enabled", False):
            self._apply_differential_privacy()
        
        # Training statistics
        self.local_steps = 0
        self.local_episodes = 0
        self.training_stats = {
            "rewards": [],
            "episode_lengths": [],
            "losses": [],
        }
    
    def _apply_differential_privacy(self):
        """Apply DP-SGD to agent's optimizer"""
        if hasattr(self.agent, "optimizer"):
            dp_params = self.dp_config.get("dp_sgd", {})
            
            # Wrap optimizer with DP (ensure numeric types)
            self.agent.optimizer = make_private(
                self.agent.optimizer,
                noise_multiplier=float(dp_params.get("noise_multiplier", 1.1)),
                max_grad_norm=float(dp_params.get("max_grad_norm", 1.0)),
                batch_size=int(self.agent.config.get("batch_size", 64)),
                target_epsilon=float(dp_params.get("target_epsilon", 10.0)) if dp_params.get("target_epsilon") else None,
                target_delta=float(dp_params.get("target_delta", 1e-5)),
            )
            
            self.logger.info("Differential privacy enabled")
    
    def train_local_epochs(
        self, num_epochs: int, episodes_per_epoch: int = 10
    ) -> Dict[str, Any]:
        """
        Train agent for specified number of epochs
        
        Args:
            num_epochs: Number of local training epochs
            episodes_per_epoch: Episodes per epoch
        
        Returns:
            Training statistics
        """
        epoch_stats = {
            "total_reward": [],
            "episode_length": [],
            "policy_loss": [],
            "value_loss": [],
        }
        
        for epoch in range(num_epochs):
            for episode in range(episodes_per_epoch):
                stats = self._run_episode()
                
                epoch_stats["total_reward"].append(stats["total_reward"])
                epoch_stats["episode_length"].append(stats["episode_length"])
                
                if "policy_loss" in stats:
                    epoch_stats["policy_loss"].append(stats["policy_loss"])
                if "value_loss" in stats:
                    epoch_stats["value_loss"].append(stats["value_loss"])
                
                self.local_episodes += 1
            
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Avg Reward: {np.mean(epoch_stats['total_reward']):.2f}"
            )
        
        # Aggregate statistics
        summary = {
            "avg_reward": np.mean(epoch_stats["total_reward"]),
            "avg_episode_length": np.mean(epoch_stats["episode_length"]),
            "episodes_trained": num_epochs * episodes_per_epoch,
        }
        
        if epoch_stats["policy_loss"]:
            summary["avg_policy_loss"] = np.mean(epoch_stats["policy_loss"])
        if epoch_stats["value_loss"]:
            summary["avg_value_loss"] = np.mean(epoch_stats["value_loss"])
        
        return summary
    
    def _run_episode(self) -> Dict[str, Any]:
        """Run single training episode"""
        state, info = self.env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        episode_length = 0
        
        while not (done or truncated):
            # Select action
            action, action_info = self.agent.select_action(state)
            
            # Take step
            next_state, reward, done, truncated, info = self.env.step(action)
            
            # Store transition
            self.agent.store_transition(
                state, action, reward, next_state, done, action_info
            )
            
            # Update counters
            episode_reward += reward
            episode_length += 1
            self.local_steps += 1
            
            state = next_state
        
        # Update agent
        try:
            update_stats = self.agent.update(next_state if not done else None)
        except TypeError:
            # fallback to backward-compatible call
            update_stats = self.agent.update()
        
        # Collect statistics
        stats = {
            "total_reward": episode_reward,
            "episode_length": episode_length,
            **update_stats,
        }
        
        return stats
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get local model parameters"""
        return self.agent.get_parameters()
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set local model parameters (from global model)"""
        self.agent.set_parameters(parameters)
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate agent performance
        
        Args:
            num_episodes: Number of evaluation episodes
        
        Returns:
            Evaluation metrics
        """
        from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
        
        eval_rewards = []
        eval_lengths = []
        y_true = []
        y_pred = []
        
        for _ in range(num_episodes):
            state, info = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0.0
            episode_length = 0
            
            while not (done or truncated):
                action, _ = self.agent.select_action(state, deterministic=True)
                next_state, reward, done, truncated, info = self.env.step(action)
                
                # Collect labels for classification metrics
                if "is_attack" in info:
                    y_true.append(1 if info["is_attack"] else 0)
                    # Predict attack if action >= 3 (heavy throttle or block)
                    # This matches environment's definition of "attack blocked"
                    y_pred.append(1 if action >= 3 else 0)
                
                episode_reward += reward
                episode_length += 1
                state = next_state
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        # Compute classification metrics if we have labels
        result = {
            "avg_reward": np.mean(eval_rewards),
            "std_reward": np.std(eval_rewards),
            "avg_length": np.mean(eval_lengths),
        }
        
        if y_true and y_pred:
            try:
                result["f1_score"] = f1_score(y_true, y_pred, zero_division=0)
                result["precision"] = precision_score(y_true, y_pred, zero_division=0)
                result["recall"] = recall_score(y_true, y_pred, zero_division=0)
                
                # Compute FPR if possible
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    result["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0
                    result["tpr"] = tp / (tp + fn) if (tp + fn) > 0 else 0
            except Exception:
                # If metrics computation fails, just skip them
                pass
        
        return result
    
    def get_privacy_spent(self) -> Optional[Dict[str, float]]:
        """Get privacy budget spent by this client"""
        if self.dp_config is None or not self.dp_config.get("enabled", False):
            return None
        
        if hasattr(self.agent.optimizer, "dp_sgd"):
            dp_sgd = self.agent.optimizer.dp_sgd
            
            # Compute actual privacy spent based on training steps
            # Estimate num_samples from environment buffer or use reasonable default
            num_samples = int(getattr(self.env, 'num_samples', 10000))
            batch_size = int(self.agent.config.get("batch_size", 64))
            
            # Get updated epsilon by calling DP-SGD's privacy computation
            if hasattr(dp_sgd, 'get_privacy_spent'):
                epsilon, _ = dp_sgd.get_privacy_spent(num_samples, batch_size)
            else:
                epsilon = dp_sgd.total_epsilon
            
            return {
                "epsilon": float(epsilon),
                "delta": float(dp_sgd.target_delta),
                "steps": int(dp_sgd.steps),
            }
        
        return None
