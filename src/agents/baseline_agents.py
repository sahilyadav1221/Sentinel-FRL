"""
Baseline Agents for Comparison
"""

import numpy as np
from typing import Dict, Tuple, Optional


class HeuristicAgent:
    """
    Simple heuristic-based agent
    Uses threshold-based rules to detect anomalies
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        threshold_method: str = "adaptive",
        **kwargs
    ):
        """
        Initialize heuristic agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            threshold_method: 'static' or 'adaptive'
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.threshold_method = threshold_method
        
        # Static thresholds (will be calibrated)
        self.flow_count_threshold = 200
        self.packet_rate_threshold = 1500
        self.byte_rate_threshold = 750000
        self.queue_threshold = 0.7
        
        # Adaptive thresholds
        self.flow_count_history = []
        self.packet_rate_history = []
        self.max_history = 1000
        
        self.update_count = 0
    
    def select_action(
        self, state: np.ndarray, deterministic: bool = True
    ) -> Tuple[int, Dict]:
        """
        Select action based on heuristics
        
        State features:
        [flow_count, packet_rate, byte_rate, avg_packet_size, 
         flow_duration_mean, flow_duration_std, port_diversity, ip_diversity,
         queue_length, queue_utilization, recent_actions...]
        """
        # Extract key features
        flow_count = state[0]
        packet_rate = state[1]
        byte_rate = state[2]
        port_diversity = state[6]
        ip_diversity = state[7]
        queue_utilization = state[9]
        
        # Update history for adaptive thresholds
        if self.threshold_method == "adaptive":
            self.flow_count_history.append(flow_count)
            self.packet_rate_history.append(packet_rate)
            
            if len(self.flow_count_history) > self.max_history:
                self.flow_count_history.pop(0)
                self.packet_rate_history.pop(0)
            
            # Update thresholds (mean + 2*std)
            if len(self.flow_count_history) >= 100:
                self.flow_count_threshold = (
                    np.mean(self.flow_count_history) +
                    2 * np.std(self.flow_count_history)
                )
                self.packet_rate_threshold = (
                    np.mean(self.packet_rate_history) +
                    2 * np.std(self.packet_rate_history)
                )
        
        # Decision logic
        anomaly_score = 0
        
        # High traffic volume
        if flow_count > self.flow_count_threshold:
            anomaly_score += 2
        if packet_rate > self.packet_rate_threshold:
            anomaly_score += 2
        if byte_rate > self.byte_rate_threshold:
            anomaly_score += 1
        
        # Queue buildup
        if queue_utilization > self.queue_threshold:
            anomaly_score += 2
        
        # Port scanning pattern (high port diversity)
        if port_diversity > 0.75:
            anomaly_score += 3
        
        # DDoS pattern (low IP diversity with high traffic)
        if ip_diversity < 0.3 and flow_count > self.flow_count_threshold:
            anomaly_score += 3
        
        # Map anomaly score to action
        # 0: no_action, 1: light_throttle, 2: medium_throttle, 
        # 3: heavy_throttle, 4: block
        if anomaly_score >= 7:
            action = 4  # Block
        elif anomaly_score >= 5:
            action = 3  # Heavy throttle
        elif anomaly_score >= 3:
            action = 2  # Medium throttle
        elif anomaly_score >= 1:
            action = 1  # Light throttle
        else:
            action = 0  # No action
        
        info = {
            "anomaly_score": anomaly_score,
            "value": 0.0,
            "log_prob": 0.0,
            "entropy": 0.0,
        }
        
        return action, info
    
    def store_transition(self, *args, **kwargs):
        """No-op for heuristic agent"""
        pass
    
    def update(self) -> Dict[str, float]:
        """No-op for heuristic agent"""
        self.update_count += 1
        return {}
    
    def get_parameters(self) -> Dict:
        """Get heuristic parameters"""
        return {
            "flow_count_threshold": self.flow_count_threshold,
            "packet_rate_threshold": self.packet_rate_threshold,
            "byte_rate_threshold": self.byte_rate_threshold,
            "queue_threshold": self.queue_threshold,
        }
    
    def set_parameters(self, parameters: Dict):
        """Set heuristic parameters"""
        for key, value in parameters.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save(self, path: str):
        """Save thresholds"""
        import json
        with open(path, 'w') as f:
            json.dump(self.get_parameters(), f)
    
    def load(self, path: str):
        """Load thresholds"""
        import json
        with open(path, 'r') as f:
            parameters = json.load(f)
            self.set_parameters(parameters)


class RandomAgent:
    """Random agent for baseline comparison"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seed: Optional[int] = None,
        **kwargs
    ):
        """Initialize random agent"""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rng = np.random.RandomState(seed)
    
    def select_action(
        self, state: np.ndarray, deterministic: bool = True
    ) -> Tuple[int, Dict]:
        """Select random action"""
        action = self.rng.randint(0, self.action_dim)
        info = {
            "value": 0.0,
            "log_prob": np.log(1.0 / self.action_dim),
            "entropy": np.log(self.action_dim),
        }
        return action, info
    
    def store_transition(self, *args, **kwargs):
        """No-op"""
        pass
    
    def update(self) -> Dict[str, float]:
        """No-op"""
        return {}
    
    def get_parameters(self) -> Dict:
        """No parameters"""
        return {}
    
    def set_parameters(self, parameters: Dict):
        """No-op"""
        pass
    
    def save(self, path: str):
        """No-op"""
        pass
    
    def load(self, path: str):
        """No-op"""
        pass


class CentralizedAgent:
    """
    Wrapper for centralized RL agent
    Trains on pooled data from all sites
    """
    
    def __init__(
        self,
        agent_type: str,
        state_dim: int,
        action_dim: int,
        config: Optional[Dict] = None,
        device: str = "cpu",
    ):
        """
        Initialize centralized agent
        
        Args:
            agent_type: 'ppo' or 'a2c'
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration
            device: Device
        """
        from .ppo_agent import PPOAgent
        from .a2c_agent import A2CAgent
        
        if agent_type.lower() == "ppo":
            self.agent = PPOAgent(state_dim, action_dim, config, device)
        elif agent_type.lower() == "a2c":
            self.agent = A2CAgent(state_dim, action_dim, config, device)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def __getattr__(self, name):
        """Delegate to wrapped agent"""
        return getattr(self.agent, name)
