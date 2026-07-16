"""
Anomaly Generator
Generates various types of network attacks
"""

import numpy as np
from typing import Dict, List, Optional
from enum import Enum


class AttackType(Enum):
    """Types of network attacks"""
    DOS = "dos"
    DDOS = "ddos"
    PORT_SCAN = "port_scan"
    SYN_FLOOD = "syn_flood"
    NONE = "none"


class AnomalyGenerator:
    """Generates attack patterns for training and evaluation"""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize anomaly generator
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        self.current_attack = AttackType.NONE
        self.attack_duration = 0
        self.attack_start_time = 0
    
    def should_inject_attack(self, time_step: int, probability: float = 0.3) -> bool:
        """
        Determine if attack should be injected at current time step
        
        Args:
            time_step: Current simulation time step
            probability: Probability of attack starting
        
        Returns:
            bool: Whether to inject attack
        """
        if self.current_attack != AttackType.NONE:
            # Check if current attack should end
            if time_step - self.attack_start_time >= self.attack_duration:
                self.current_attack = AttackType.NONE
                return False
            return True
        
        # Start new attack with given probability
        if self.rng.random() < probability:
            return True
        
        return False
    
    def generate_attack(
        self, attack_type: Optional[str] = None, intensity: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Generate attack parameters
        
        Args:
            attack_type: Specific attack type (random if None)
            intensity: Attack intensity 0-1 (random if None)
        
        Returns:
            dict: Attack parameters
        """
        if attack_type is None:
            attack_types = [AttackType.DOS, AttackType.DDOS, 
                          AttackType.PORT_SCAN, AttackType.SYN_FLOOD]
            self.current_attack = self.rng.choice(attack_types)
        else:
            self.current_attack = AttackType(attack_type)
        
        if intensity is None:
            intensity = self.rng.uniform(0.1, 0.9)
        
        # Attack duration: 10-100 time steps
        self.attack_duration = self.rng.randint(10, 100)
        self.attack_start_time = 0  # Will be set externally
        
        return {
            "type": self.current_attack,
            "intensity": intensity,
            "duration": self.attack_duration,
            "characteristics": self._get_attack_characteristics(
                self.current_attack, intensity
            ),
        }
    
    def _get_attack_characteristics(
        self, attack_type: AttackType, intensity: float
    ) -> Dict[str, float]:
        """Get characteristic parameters for specific attack type"""
        
        if attack_type == AttackType.DOS:
            return {
                "flow_multiplier": 1 + intensity * 10,
                "packet_multiplier": 1 + intensity * 5,
                "byte_multiplier": 1 + intensity * 5,
                "ip_diversity_reduction": intensity * 0.8,
                "port_diversity_change": 0.0,
                "packet_size_change": 0.0,
            }
        
        elif attack_type == AttackType.DDOS:
            return {
                "flow_multiplier": 1 + intensity * 15,
                "packet_multiplier": 1 + intensity * 8,
                "byte_multiplier": 1 + intensity * 8,
                "ip_diversity_reduction": intensity * 0.3,  # Higher diversity
                "port_diversity_change": 0.0,
                "packet_size_change": 0.0,
            }
        
        elif attack_type == AttackType.PORT_SCAN:
            return {
                "flow_multiplier": 1 + intensity * 20,
                "packet_multiplier": 1 + intensity * 0.5,
                "byte_multiplier": 1 + intensity * 0.3,
                "ip_diversity_reduction": 0.0,
                "port_diversity_change": intensity * 0.7,  # Increase diversity
                "packet_size_change": -intensity * 0.5,  # Smaller packets
            }
        
        elif attack_type == AttackType.SYN_FLOOD:
            return {
                "flow_multiplier": 1 + intensity * 8,
                "packet_multiplier": 1 + intensity * 10,
                "byte_multiplier": 1 + intensity * 2,
                "ip_diversity_reduction": intensity * 0.6,
                "port_diversity_change": 0.0,
                "packet_size_change": -intensity * 0.7,  # Much smaller packets
            }
        
        return {}
    
    def get_detection_difficulty(self) -> float:
        """
        Get difficulty of detecting current attack (for evaluation)
        
        Returns:
            float: Difficulty score 0-1 (1 = hardest)
        """
        if self.current_attack == AttackType.NONE:
            return 0.0
        
        # Port scans and low-intensity attacks are harder to detect
        difficulty_map = {
            AttackType.DOS: 0.3,
            AttackType.DDOS: 0.4,
            AttackType.PORT_SCAN: 0.8,
            AttackType.SYN_FLOOD: 0.5,
        }
        
        return difficulty_map.get(self.current_attack, 0.5)
    
    def reset(self):
        """Reset generator state"""
        self.current_attack = AttackType.NONE
        self.attack_duration = 0
        self.attack_start_time = 0
