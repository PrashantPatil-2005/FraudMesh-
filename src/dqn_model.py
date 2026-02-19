import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Deep Q-Network for fraud response action selection.
    
    Architecture:
    Input (state_dim=8) -> FC(128) -> ReLU -> Dropout
                        -> FC(128) -> ReLU -> Dropout
                        -> FC(64) -> ReLU
                        -> FC(num_actions=5)
    """
    
    def __init__(self, state_dim, num_actions, hidden_dim=128):
        super(QNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        return self.network(state)
    
    def get_action_values(self, state):
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            q_values = self.forward(state)
            return q_values.squeeze(0)
