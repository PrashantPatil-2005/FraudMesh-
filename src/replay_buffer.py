import numpy as np
import random
from collections import deque
from src.config import DQN_BUFFER_SIZE


class ReplayBuffer:
    """
    Fixed-size replay buffer for storing and sampling experiences.
    Each experience is a tuple: (state, action, reward, next_state, done)
    """
    
    def __init__(self, capacity=DQN_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([e[0] for e in batch], dtype=np.float32)
        actions = np.array([e[1] for e in batch], dtype=np.int64)
        rewards = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.array([e[3] for e in batch], dtype=np.float32)
        dones = np.array([e[4] for e in batch], dtype=np.bool_)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
    def is_ready(self, min_size):
        return len(self.buffer) >= min_size
