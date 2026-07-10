"""
Test Environment
"""

import pytest
import numpy as np
from src.environment.gateway_env import GatewayAnomalyEnv
from src.environment.traffic_simulator import TrafficSimulator, TrafficPattern


def test_environment_creation():
    """Test that environment can be created"""
    env = GatewayAnomalyEnv(pattern="normal_moderate", episode_length=100)
    assert env is not None
    assert env.observation_space.shape[0] > 0
    assert env.action_space.n == 5


def test_environment_reset():
    """Test environment reset"""
    env = GatewayAnomalyEnv(pattern="normal_moderate")
    obs, info = env.reset()
    
    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)


def test_environment_step():
    """Test environment step"""
    env = GatewayAnomalyEnv(pattern="normal_moderate")
    obs, info = env.reset()
    
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    assert next_obs.shape == env.observation_space.shape
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_traffic_simulator():
    """Test traffic simulator"""
    sim = TrafficSimulator(pattern=TrafficPattern.NORMAL_MODERATE, seed=42)
    
    flow_stats, is_attack, attack_type = sim.step()
    
    assert flow_stats.flow_count > 0
    assert flow_stats.packet_rate >= 0
    assert isinstance(is_attack, bool)


def test_episode_completion():
    """Test full episode"""
    env = GatewayAnomalyEnv(pattern="normal_moderate", episode_length=10)
    obs, info = env.reset()
    
    total_reward = 0
    steps = 0
    done = False
    truncated = False
    
    while not (done or truncated):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
    
    assert steps == 10  # Should match episode_length
    stats = env.get_episode_statistics()
    assert 'f1_score' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
