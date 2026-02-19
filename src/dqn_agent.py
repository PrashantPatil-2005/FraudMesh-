import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.dqn_model import QNetwork
from src.replay_buffer import ReplayBuffer
from src.config import (
    RL_STATE_DIM, RL_NUM_ACTIONS, DQN_HIDDEN_DIM,
    DQN_LEARNING_RATE, DQN_GAMMA, DQN_TAU, DQN_BATCH_SIZE,
    DQN_BUFFER_SIZE, DQN_MIN_BUFFER_SIZE,
    EPSILON_START, EPSILON_END, EPSILON_DECAY,
    RL_CHECKPOINT_PATH
)


class DQNAgent:
    """
    Deep Q-Network Agent with experience replay and target network.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.q_network = QNetwork(
            state_dim=RL_STATE_DIM,
            num_actions=RL_NUM_ACTIONS,
            hidden_dim=DQN_HIDDEN_DIM
        ).to(self.device)
        
        self.target_network = QNetwork(
            state_dim=RL_STATE_DIM,
            num_actions=RL_NUM_ACTIONS,
            hidden_dim=DQN_HIDDEN_DIM
        ).to(self.device)
        
        # Copy weights from q_network to target_network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=DQN_LEARNING_RATE
        )
        self.criterion = nn.SmoothL1Loss()  # Huber loss
        
        # Replay buffer
        self.buffer = ReplayBuffer(capacity=DQN_BUFFER_SIZE)
        
        # Exploration
        self.epsilon = EPSILON_START
        self.epsilon_end = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY
        
        # Tracking
        self.train_step_count = 0
        self.losses = []
    
    def select_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, RL_NUM_ACTIONS)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            q_values = self.q_network.get_action_values(state_tensor)
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
    
    def learn(self):
        if not self.buffer.is_ready(DQN_MIN_BUFFER_SIZE):
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(DQN_BATCH_SIZE)
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        
        # Current Q-values: Q(s, a)
        current_q_values = self.q_network(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q = next_q_values.max(dim=1)[0]
            target_q = rewards + DQN_GAMMA * max_next_q * (~dones).float()
        
        # Loss and update
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Soft-update target network
        self._soft_update_target()
        
        self.train_step_count += 1
        loss_val = loss.item()
        self.losses.append(loss_val)
        
        return loss_val
    
    def _soft_update_target(self):
        for target_param, q_param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                DQN_TAU * q_param.data + (1.0 - DQN_TAU) * target_param.data
            )
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path=None):
        if path is None:
            path = RL_CHECKPOINT_PATH
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step_count': self.train_step_count,
            'losses': self.losses
        }, path)
    
    def load(self, path=None):
        if path is None:
            path = RL_CHECKPOINT_PATH
        checkpoint = torch.load(path, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.train_step_count = checkpoint['train_step_count']
        self.losses = checkpoint['losses']
