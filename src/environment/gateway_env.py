"""
Gateway Anomaly Detection Environment
OpenAI Gym-compatible environment for RL training
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

from .traffic_simulator import TrafficSimulator, TrafficPattern, FlowStats
from .anomaly_generator import AnomalyGenerator, AttackType


class GatewayAnomalyEnv(gym.Env):
    """
    Environment for training RL agents to detect and respond to network anomalies
    
    State: Flow statistics, queue metrics, recent action history
    Action: Throttle/block levels (discrete: 5 actions)
    Reward: -benign_loss - attack_pass - latency_penalty
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        pattern: str = "normal_moderate",
        episode_length: int = 1000,
        seed: Optional[int] = None,
        reward_weights: Optional[Dict[str, float]] = None,
        queue_capacity: int = 10000,
        action_history_length: int = 5,
    ):
        """
        Initialize environment
        
        Args:
            pattern: Traffic pattern type
            episode_length: Maximum steps per episode
            seed: Random seed
            reward_weights: Weights for reward components
            queue_capacity: Maximum queue size
            action_history_length: Number of recent actions to include in state
        """
        super().__init__()
        
        self.episode_length = episode_length
        self.queue_capacity = queue_capacity
        self.action_history_length = action_history_length
        
        # Initialize components
        pattern_enum = TrafficPattern(pattern)
        self.traffic_sim = TrafficSimulator(pattern=pattern_enum, seed=seed)
        self.anomaly_gen = AnomalyGenerator(seed=seed)
        
        # Reward weights
        if reward_weights is None:
            reward_weights = {
                "benign_loss": -1.0,
                "attack_pass": -2.0,
                "latency": -0.5,
                "correct_action": 0.1,
            }
        self.reward_weights = reward_weights
        
        # Action space: [no_action, light_throttle, medium_throttle, heavy_throttle, block]
        self.action_space = spaces.Discrete(5)
        self.action_names = [
            "no_action",
            "light_throttle",
            "medium_throttle", 
            "heavy_throttle",
            "block"
        ]
        
        # State space: flow stats (8) + queue (2) + recent actions (5)
        state_dim = 8 + 2 + action_history_length
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Internal state
        self.current_step = 0
        self.queue_length = 0
        self.action_history = deque(maxlen=action_history_length)
        self.episode_stats = {
            "total_reward": 0.0,
            "benign_blocked": 0,
            "benign_passed": 0,
            "attack_blocked": 0,
            "attack_passed": 0,
        }
        
        self.rng = np.random.RandomState(seed)
    
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            self.traffic_sim.rng = self.rng
            self.anomaly_gen.rng = self.rng
        
        self.current_step = 0
        self.queue_length = 0
        self.action_history.clear()
        for _ in range(self.action_history_length):
            self.action_history.append(0)
        
        self.episode_stats = {
            "total_reward": 0.0,
            "benign_blocked": 0,
            "benign_passed": 0,
            "attack_blocked": 0,
            "attack_passed": 0,
        }
        
        self.traffic_sim.reset()
        self.anomaly_gen.reset()
        
        # Get initial observation
        flow_stats, is_attack, attack_type = self.traffic_sim.step()
        obs = self._build_observation(flow_stats)
        info = self._build_info(flow_stats, is_attack, attack_type)
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Action to take (0-4)
        
        Returns:
            observation: Next state
            reward: Reward for this step
            terminated: Whether episode ended naturally
            truncated: Whether episode was cut off
            info: Additional information
        """
        self.current_step += 1
        self.action_history.append(action)
        
        # Simulate traffic
        flow_stats, is_attack, attack_type = self.traffic_sim.step()
        
        # Apply action and compute consequences
        benign_loss, attack_pass, latency = self._apply_action(
            action, flow_stats, is_attack
        )
        
        # Update queue
        self._update_queue(flow_stats, action)
        
        # Calculate reward
        reward = self._calculate_reward(benign_loss, attack_pass, latency, is_attack, action)
        
        # Update episode statistics
        self.episode_stats["total_reward"] += reward
        if is_attack:
            if action >= 3:  # Heavy throttle or block
                self.episode_stats["attack_blocked"] += 1
            else:
                self.episode_stats["attack_passed"] += 1
        else:
            if action >= 3:
                self.episode_stats["benign_blocked"] += 1
            else:
                self.episode_stats["benign_passed"] += 1
        
        # Build next observation
        obs = self._build_observation(flow_stats)
        
        # Check if episode is done
        terminated = False
        truncated = self.current_step >= self.episode_length
        
        # Build info dict
        info = self._build_info(flow_stats, is_attack, attack_type)
        info.update({
            "benign_loss": benign_loss,
            "attack_pass": attack_pass,
            "latency": latency,
            "action_name": self.action_names[action],
        })
        
        return obs, reward, terminated, truncated, info
    
    def _build_observation(self, flow_stats: FlowStats) -> np.ndarray:
        """Build observation vector from current state"""
        # Flow statistics (8 features)
        flow_features = flow_stats.to_vector()
        
        # Queue metrics (2 features)
        queue_features = np.array([
            self.queue_length,
            self.queue_length / self.queue_capacity,  # Utilization
        ], dtype=np.float32)
        
        # Recent actions (5 features)
        action_features = np.array(list(self.action_history), dtype=np.float32)
        
        # Concatenate all features
        obs = np.concatenate([flow_features, queue_features, action_features])
        
        return obs
    
    def _apply_action(
        self, action: int, flow_stats: FlowStats, is_attack: bool
    ) -> Tuple[float, float, float]:
        """
        Apply action and compute consequences
        
        Returns:
            benign_loss: Fraction of benign traffic blocked (0-1)
            attack_pass: Fraction of attack traffic that passed (0-1)
            latency: Latency penalty (0-1)
        """
        # Action effectiveness
        throttle_rates = {
            0: 0.0,   # No action
            1: 0.2,   # Light throttle
            2: 0.5,   # Medium throttle
            3: 0.8,   # Heavy throttle
            4: 1.0,   # Block
        }
        
        throttle = throttle_rates[action]
        
        # Compute benign loss (false positives)
        if not is_attack:
            benign_loss = throttle
            attack_pass = 0.0
        else:
            # Some benign traffic mixed with attack
            benign_ratio = 0.1  # Assume 10% benign even during attack
            benign_loss = throttle * benign_ratio
            
            # Attack pass rate (false negatives)
            attack_pass = max(0, 1.0 - throttle - self.rng.uniform(0, 0.1))
        
        # Latency increases with queue length and throttling
        queue_latency = self.queue_length / self.queue_capacity
        throttle_latency = throttle * 0.3  # Throttling adds latency
        latency = queue_latency + throttle_latency
        
        return benign_loss, attack_pass, latency
    
    def _update_queue(self, flow_stats: FlowStats, action: int):
        """Update queue based on incoming traffic and action"""
        # Incoming traffic
        incoming = flow_stats.flow_count
        
        # Action reduces incoming traffic
        throttle_rates = {0: 0.0, 1: 0.2, 2: 0.5, 3: 0.8, 4: 1.0}
        throttle = throttle_rates[action]
        incoming = int(incoming * (1 - throttle))
        
        # Process some queue
        processing_rate = 100  # Flows processed per step
        self.queue_length = max(0, self.queue_length - processing_rate)
        
        # Add incoming traffic
        self.queue_length = min(
            self.queue_capacity, 
            self.queue_length + incoming
        )
    
    def _calculate_reward(
        self,
        benign_loss: float,
        attack_pass: float,
        latency: float,
        is_attack: bool,
        action: int
    ) -> float:
        """Calculate reward based on consequences"""
        reward = (
            self.reward_weights["benign_loss"] * benign_loss +
            self.reward_weights["attack_pass"] * attack_pass +
            self.reward_weights["latency"] * latency
        )
        
        # Bonus for correct action
        if is_attack and action >= 3:  # Correctly blocking attack
            reward += self.reward_weights["correct_action"]
        elif not is_attack and action <= 1:  # Correctly allowing benign
            reward += self.reward_weights["correct_action"]
        
        return reward
    
    def _build_info(
        self, flow_stats: FlowStats, is_attack: bool, attack_type: Optional[str]
    ) -> Dict[str, Any]:
        """Build info dictionary"""
        ground_truth = self.traffic_sim.get_ground_truth()
        
        return {
            "step": self.current_step,
            "is_attack": is_attack,
            "attack_type": attack_type,
            "attack_intensity": ground_truth.get("attack_intensity", 0.0),
            "queue_length": self.queue_length,
            "queue_utilization": self.queue_length / self.queue_capacity,
            "flow_count": flow_stats.flow_count,
            "packet_rate": flow_stats.packet_rate,
        }
    
    def render(self):
        """Render environment (optional)"""
        if self.current_step % 100 == 0:
            print(f"Step: {self.current_step}/{self.episode_length}")
            print(f"Queue: {self.queue_length}/{self.queue_capacity}")
            print(f"Stats: {self.episode_stats}")
    
    def close(self):
        """Clean up resources"""
        pass
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """Get statistics for completed episode"""
        total_interactions = (
            self.episode_stats["benign_blocked"] +
            self.episode_stats["benign_passed"] +
            self.episode_stats["attack_blocked"] +
            self.episode_stats["attack_passed"]
        )
        
        if total_interactions == 0:
            return self.episode_stats
        
        stats = self.episode_stats.copy()
        
        # Calculate metrics
        true_positives = stats["attack_blocked"]
        false_positives = stats["benign_blocked"]
        true_negatives = stats["benign_passed"]
        false_negatives = stats["attack_passed"]
        
        # Precision, Recall, F1
        precision = true_positives / max(1, true_positives + false_positives)
        recall = true_positives / max(1, true_positives + false_negatives)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        
        # False Positive Rate
        fpr = false_positives / max(1, false_positives + true_negatives)
        
        stats.update({
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "fpr": fpr,
        })
        
        return stats
