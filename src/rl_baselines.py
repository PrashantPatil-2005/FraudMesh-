import numpy as np
from src.config import RL_NUM_ACTIONS, RL_ACTIONS


class RandomPolicy:
    """Baseline: picks a random action for every transaction."""
    
    def __init__(self):
        self.name = "Random Policy"
    
    def select_action(self, state, training=False):
        return np.random.randint(0, RL_NUM_ACTIONS)


class RuleBasedPolicy:
    """
    Baseline: uses simple threshold rules based on fraud score.
    
    Rules:
        fraud_score < 0.2  -> APPROVE (0)
        fraud_score < 0.4  -> SOFT_BLOCK (1)
        fraud_score < 0.6  -> FLAG_REVIEW (3)
        fraud_score < 0.8  -> HARD_BLOCK (2)
        fraud_score >= 0.8 -> FREEZE_ACCOUNT (4)
    """
    
    def __init__(self):
        self.name = "Rule-Based Policy"
    
    def select_action(self, state, training=False):
        fraud_score = state[0]
        
        if fraud_score < 0.2:
            return 0    # APPROVE
        elif fraud_score < 0.4:
            return 1    # SOFT_BLOCK
        elif fraud_score < 0.6:
            return 3    # FLAG_REVIEW
        elif fraud_score < 0.8:
            return 2    # HARD_BLOCK
        else:
            return 4    # FREEZE_ACCOUNT


class AlwaysApprovePolicy:
    """Baseline: approves every transaction."""
    
    def __init__(self):
        self.name = "Always Approve"
    
    def select_action(self, state, training=False):
        return 0


class AlwaysBlockPolicy:
    """Baseline: blocks every transaction."""
    
    def __init__(self):
        self.name = "Always Block"
    
    def select_action(self, state, training=False):
        return 2
