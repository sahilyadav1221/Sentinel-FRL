"""
Neural Network Architectures for RL Agents
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import numpy as np


class ActorCriticNetwork(nn.Module):
    """
    Shared Actor-Critic network with separate heads
    Used by both PPO and A2C
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: List[int] = [256, 128, 64],
        activation: str = "relu",
        use_lstm: bool = False,
        lstm_hidden_size: int = 128,
    ):
        """
        Initialize Actor-Critic network
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_layers: List of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'elu')
            use_lstm: Whether to use LSTM layer
            lstm_hidden_size: Size of LSTM hidden state
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_lstm = use_lstm
        self.lstm_hidden_size = lstm_hidden_size
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Shared feature extraction layers
        layers = []
        input_dim = state_dim
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(self.activation)
            layers.append(nn.LayerNorm(hidden_size))
            input_dim = hidden_size
        
        self.shared_network = nn.Sequential(*layers)
        
        # LSTM layer (optional)
        if use_lstm:
            self.lstm = nn.LSTM(input_dim, lstm_hidden_size, batch_first=True)
            input_dim = lstm_hidden_size
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_layers[-1]),
            self.activation,
            nn.Linear(hidden_layers[-1], action_dim)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_layers[-1]),
            self.activation,
            nn.Linear(hidden_layers[-1], 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function"""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
        }
        return activations.get(activation.lower(), nn.ReLU())
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self,
        state: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through network
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            lstm_state: Previous LSTM hidden state (if using LSTM)
        
        Returns:
            action_logits: Logits for action distribution [batch_size, action_dim]
            value: State value estimate [batch_size, 1]
            new_lstm_state: Updated LSTM state (if using LSTM)
        """
        # Extract shared features
        features = self.shared_network(state)
        
        # Apply LSTM if enabled
        new_lstm_state = None
        if self.use_lstm:
            if lstm_state is None:
                batch_size = state.shape[0]
                device = state.device
                h0 = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
                c0 = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
                lstm_state = (h0, c0)
            
            # Add sequence dimension if needed
            if features.dim() == 2:
                features = features.unsqueeze(1)  # [batch, 1, features]
            
            features, new_lstm_state = self.lstm(features, lstm_state)
            features = features.squeeze(1)  # Remove sequence dimension
        
        # Actor and Critic outputs
        action_logits = self.actor(features)
        value = self.critic(features)
        
        return action_logits, value, new_lstm_state
    
    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, value, log probability, and entropy
        
        Args:
            state: Input state
            action: Action (if evaluating specific action)
            lstm_state: LSTM hidden state
        
        Returns:
            action: Selected action
            log_prob: Log probability of action
            entropy: Entropy of action distribution
            value: State value
        """
        action_logits, value, new_lstm_state = self.forward(state, lstm_state)
        
        # Create categorical distribution
        probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        # Sample action if not provided
        if action is None:
            action = dist.sample()
        
        # Calculate log probability and entropy
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get value estimate for state"""
        _, value, _ = self.forward(state)
        return value.squeeze(-1)


class DuelingNetwork(nn.Module):
    """
    Dueling network architecture (alternative for Q-learning based methods)
    Separates value and advantage functions
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: List[int] = [256, 128],
        activation: str = "relu",
    ):
        """Initialize dueling network"""
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        activation_fn = nn.ReLU() if activation == "relu" else nn.Tanh()
        
        # Shared feature extraction
        layers = []
        input_dim = state_dim
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(activation_fn)
            input_dim = hidden_size
        
        self.features = nn.Sequential(*layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_layers[-1] // 2),
            activation_fn,
            nn.Linear(hidden_layers[-1] // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_layers[-1] // 2),
            activation_fn,
            nn.Linear(hidden_layers[-1] // 2, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        """
        features = self.features(state)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
        
        return q_values


def create_network(
    network_type: str,
    state_dim: int,
    action_dim: int,
    config: dict
) -> nn.Module:
    """
    Factory function to create networks
    
    Args:
        network_type: Type of network ('actor_critic', 'dueling')
        state_dim: State space dimension
        action_dim: Action space dimension
        config: Network configuration
    
    Returns:
        Neural network module
    """
    if network_type == "actor_critic":
        return ActorCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=config.get("hidden_layers", [256, 128, 64]),
            activation=config.get("activation", "relu"),
            use_lstm=config.get("use_lstm", False),
            lstm_hidden_size=config.get("lstm_hidden_size", 128),
        )
    elif network_type == "dueling":
        return DuelingNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=config.get("hidden_layers", [256, 128]),
            activation=config.get("activation", "relu"),
        )
    else:
        raise ValueError(f"Unknown network type: {network_type}")
