"""
Traffic Simulator for Network Gateway
Simulates normal and attack traffic patterns
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class TrafficPattern(Enum):
    """Types of traffic patterns for different sites"""
    NORMAL_LIGHT = "normal_light"
    NORMAL_MODERATE = "normal_moderate"
    NORMAL_HEAVY = "normal_heavy"
    MIXED_LIGHT_ATTACK = "mixed_light_attack"
    MIXED_HEAVY_ATTACK = "mixed_heavy_attack"


@dataclass
class FlowStats:
    """Statistics for network flows"""
    flow_count: int
    packet_rate: float
    byte_rate: float
    avg_packet_size: float
    flow_duration_mean: float
    flow_duration_std: float
    port_diversity: float
    ip_diversity: float
    protocol_distribution: Dict[str, float]
    
    def to_vector(self) -> np.ndarray:
        """Convert flow stats to feature vector"""
        return np.array([
            self.flow_count,
            self.packet_rate,
            self.byte_rate,
            self.avg_packet_size,
            self.flow_duration_mean,
            self.flow_duration_std,
            self.port_diversity,
            self.ip_diversity,
        ], dtype=np.float32)


class TrafficSimulator:
    """Simulates network traffic with configurable patterns"""
    
    def __init__(
        self,
        pattern: TrafficPattern = TrafficPattern.NORMAL_MODERATE,
        seed: Optional[int] = None,
        base_flow_rate: float = 100.0,
        attack_probability: float = 0.3,
    ):
        """
        Initialize traffic simulator
        
        Args:
            pattern: Traffic pattern type for this site
            seed: Random seed for reproducibility
            base_flow_rate: Base number of flows per time step
            attack_probability: Probability of attack occurring
        """
        self.pattern = pattern
        self.rng = np.random.RandomState(seed)
        self.base_flow_rate = base_flow_rate
        self.attack_probability = attack_probability
        
        self.time_step = 0
        self.is_under_attack = False
        self.attack_type = None
        self.attack_intensity = 0.0
        
        # Pattern-specific parameters
        self._set_pattern_parameters()
    
    def _set_pattern_parameters(self):
        """Set parameters based on traffic pattern"""
        if self.pattern == TrafficPattern.NORMAL_LIGHT:
            self.base_flow_rate *= 0.5
            self.attack_probability = 0.0
            self.attack_stop_probability = 1.0  # Stop immediately (no attacks)
        elif self.pattern == TrafficPattern.NORMAL_MODERATE:
            self.base_flow_rate *= 1.0
            self.attack_probability = 0.0
            self.attack_stop_probability = 1.0
        elif self.pattern == TrafficPattern.NORMAL_HEAVY:
            self.base_flow_rate *= 2.0
            self.attack_probability = 0.0
            self.attack_stop_probability = 1.0
        elif self.pattern == TrafficPattern.MIXED_LIGHT_ATTACK:
            self.base_flow_rate *= 0.5
            self.attack_probability = 0.3  # 30% chance to start
            self.attack_stop_probability = 0.7  # 70% chance to stop (balances to ~30% attack time)
        elif self.pattern == TrafficPattern.MIXED_HEAVY_ATTACK:
            self.base_flow_rate *= 2.0
            self.attack_probability = 0.5  # 50% chance to start
            self.attack_stop_probability = 0.5  # 50% chance to stop (balances to ~50% attack time)
    
    def step(self) -> Tuple[FlowStats, bool, Optional[str]]:
        """
        Simulate one time step of traffic
        
        Returns:
            flow_stats: Statistics about current flows
            is_attack: Whether current traffic contains attacks
            attack_type: Type of attack if present
        """
        self.time_step += 1
        
        # Determine if attack should start/continue/stop
        if not self.is_under_attack:
            # Try to start attack based on attack probability
            if self.rng.random() < self.attack_probability:
                self._start_attack()
        else:
            # Try to stop attack based on pattern-specific stop probability
            # This balances start/stop to achieve desired attack rate
            if self.rng.random() < self.attack_stop_probability:
                self._stop_attack()
        
        # Generate flow statistics
        flow_stats = self._generate_flow_stats()
        
        return flow_stats, self.is_under_attack, self.attack_type
    
    def _start_attack(self):
        """Initiate an attack"""
        self.is_under_attack = True
        attack_types = ["dos", "ddos", "port_scan", "syn_flood"]
        self.attack_type = self.rng.choice(attack_types)
        self.attack_intensity = self.rng.uniform(0.1, 0.9)
    
    def _stop_attack(self):
        """Stop ongoing attack"""
        self.is_under_attack = False
        self.attack_type = None
        self.attack_intensity = 0.0
    
    def _generate_flow_stats(self) -> FlowStats:
        """Generate flow statistics based on current state"""
        # Base normal traffic
        flow_count = max(1, int(self.rng.poisson(self.base_flow_rate)))
        packet_rate = self.rng.normal(1000, 200)
        byte_rate = packet_rate * self.rng.normal(500, 100)
        avg_packet_size = byte_rate / max(1, packet_rate)
        
        # Normal flow characteristics
        flow_duration_mean = self.rng.exponential(10.0)
        flow_duration_std = self.rng.exponential(5.0)
        port_diversity = self.rng.uniform(0.3, 0.7)
        ip_diversity = self.rng.uniform(0.5, 0.9)
        
        # Modify stats if under attack
        if self.is_under_attack:
            flow_count, packet_rate, byte_rate = self._apply_attack_effects(
                flow_count, packet_rate, byte_rate
            )
            
            if self.attack_type == "port_scan":
                port_diversity = self.rng.uniform(0.8, 0.99)
            elif self.attack_type in ["dos", "ddos"]:
                ip_diversity = self.rng.uniform(0.1, 0.3)
                flow_duration_mean *= 0.1
            elif self.attack_type == "syn_flood":
                packet_rate *= 2.0
                avg_packet_size *= 0.5
        
        # Add noise
        packet_rate = max(0, packet_rate)
        byte_rate = max(0, byte_rate)
        avg_packet_size = byte_rate / max(1, packet_rate)
        
        protocol_distribution = {
            "tcp": self.rng.dirichlet([5, 2, 1])[0],
            "udp": self.rng.dirichlet([5, 2, 1])[1],
            "other": self.rng.dirichlet([5, 2, 1])[2],
        }
        
        return FlowStats(
            flow_count=flow_count,
            packet_rate=packet_rate,
            byte_rate=byte_rate,
            avg_packet_size=avg_packet_size,
            flow_duration_mean=flow_duration_mean,
            flow_duration_std=flow_duration_std,
            port_diversity=port_diversity,
            ip_diversity=ip_diversity,
            protocol_distribution=protocol_distribution,
        )
    
    def _apply_attack_effects(
        self, flow_count: int, packet_rate: float, byte_rate: float
    ) -> Tuple[int, float, float]:
        """Apply attack-specific effects on traffic statistics"""
        intensity = self.attack_intensity
        
        if self.attack_type in ["dos", "ddos"]:
            # Increase traffic volume dramatically
            flow_count = int(flow_count * (1 + intensity * 10))
            packet_rate *= (1 + intensity * 5)
            byte_rate *= (1 + intensity * 5)
        
        elif self.attack_type == "port_scan":
            # Many flows, but low data rate
            flow_count = int(flow_count * (1 + intensity * 20))
            packet_rate *= (1 + intensity * 0.5)
        
        elif self.attack_type == "syn_flood":
            # High packet rate, small packets
            flow_count = int(flow_count * (1 + intensity * 8))
            packet_rate *= (1 + intensity * 10)
            byte_rate *= (1 + intensity * 2)
        
        return flow_count, packet_rate, byte_rate
    
    def reset(self):
        """Reset simulator to initial state"""
        self.time_step = 0
        self.is_under_attack = False
        self.attack_type = None
        self.attack_intensity = 0.0
    
    def get_ground_truth(self) -> Dict[str, any]:
        """Get ground truth about current state (for evaluation)"""
        return {
            "is_attack": self.is_under_attack,
            "attack_type": self.attack_type,
            "attack_intensity": self.attack_intensity,
            "time_step": self.time_step,
        }
