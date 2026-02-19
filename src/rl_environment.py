import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from src.config import (
    RL_ACTIONS, RL_NUM_ACTIONS, RL_REWARDS, RL_STATE_DIM,
    RL_MAX_STEPS_PER_EPISODE, TARGET_COL, AMOUNT_COL,
    CARD_COL, MERCHANT_NODE_COL, TIME_COL
)


class FraudResponseEnv(gym.Env):
    """
    Custom Gymnasium environment for fraud response decisions.
    
    Observation Space: Box(0, 1, shape=(8,))
    Action Space: Discrete(5)
        0 = APPROVE, 1 = SOFT_BLOCK, 2 = HARD_BLOCK,
        3 = FLAG_REVIEW, 4 = FREEZE_ACCOUNT
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, transactions_df, fraud_scores, render_mode=None):
        super(FraudResponseEnv, self).__init__()
        
        self.transactions = transactions_df.reset_index(drop=True)
        self.fraud_scores = fraud_scores.astype(np.float32)
        self.num_transactions = len(self.transactions)
        self.render_mode = render_mode
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(RL_STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(RL_NUM_ACTIONS)
        
        # Precompute features
        self._precompute_features()
        
        # Episode tracking
        self.current_step = 0
        self.current_indices = None
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_outcomes = []
    
    def _precompute_features(self):
        """Precompute all state features for all transactions."""
        df = self.transactions
        
        # Feature 1: fraud_score (from GNN)
        self.feat_fraud_score = np.clip(self.fraud_scores, 0.0, 1.0)
        
        # Feature 2: amount_normalized (log transform)
        amounts = df[AMOUNT_COL].values.astype(np.float32)
        log_amounts = np.log1p(amounts)
        amt_min, amt_max = log_amounts.min(), log_amounts.max()
        if amt_max > amt_min:
            self.feat_amount = (log_amounts - amt_min) / (amt_max - amt_min)
        else:
            self.feat_amount = np.zeros_like(log_amounts)
        
        # Feature 3: hour_of_day (normalized 0-1)
        times = df[TIME_COL].values.astype(np.float64)
        hours = (times % 86400) / 3600.0
        self.feat_hour = (hours / 24.0).astype(np.float32)
        
        # Feature 4: is_high_amount
        amount_90th = np.percentile(amounts, 90)
        self.feat_high_amount = (amounts > amount_90th).astype(np.float32)
        
        # Feature 5: card_frequency (normalized)
        card_counts = df[CARD_COL].value_counts()
        card_freq = df[CARD_COL].map(card_counts).values.astype(np.float32)
        freq_max = card_freq.max()
        self.feat_card_freq = card_freq / freq_max if freq_max > 0 else np.zeros_like(card_freq)
        
        # Feature 6: merchant_risk
        merchant_fraud_rate = df.groupby(MERCHANT_NODE_COL)[TARGET_COL].mean()
        merchant_risk = df[MERCHANT_NODE_COL].map(merchant_fraud_rate).values.astype(np.float32)
        overall_fraud_rate = df[TARGET_COL].mean()
        merchant_risk = np.nan_to_num(merchant_risk, nan=overall_fraud_rate)
        self.feat_merchant_risk = np.clip(merchant_risk, 0.0, 1.0)
        
        # Feature 7: amount_zscore (normalized 0-1)
        amt_mean, amt_std = amounts.mean(), amounts.std()
        if amt_std > 0:
            zscores = (amounts - amt_mean) / amt_std
        else:
            zscores = np.zeros_like(amounts)
        zscores = np.clip(zscores, -3.0, 3.0)
        self.feat_zscore = ((zscores + 3.0) / 6.0).astype(np.float32)
        
        # Feature 8: velocity (transactions in nearby window)
        velocity = np.zeros(self.num_transactions, dtype=np.float32)
        cards = df[CARD_COL].values
        window = 100
        
        # Optimized velocity computation using vectorized card matching
        print("    Computing velocity features (this may take a minute)...")
        for i in range(self.num_transactions):
            start = max(0, i - window)
            end = min(self.num_transactions, i + window)
            same_card_count = np.sum(cards[start:end] == cards[i])
            velocity[i] = same_card_count
        
        vel_max = velocity.max()
        self.feat_velocity = velocity / vel_max if vel_max > 0 else np.zeros_like(velocity)
        
        # Stack all features
        self.all_states = np.column_stack([
            self.feat_fraud_score,
            self.feat_amount,
            self.feat_hour,
            self.feat_high_amount,
            self.feat_card_freq,
            self.feat_merchant_risk,
            self.feat_zscore,
            self.feat_velocity
        ]).astype(np.float32)
        
        self.all_labels = df[TARGET_COL].values.astype(np.int32)
        
        print(f"  Environment initialized:")
        print(f"    Transactions: {self.num_transactions:,}")
        print(f"    Fraud rate: {self.all_labels.mean()*100:.2f}%")
        print(f"    State dim: {self.all_states.shape[1]}")
        print(f"    Actions: {RL_NUM_ACTIONS}")
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_indices = self.np_random.choice(
            self.num_transactions,
            size=min(RL_MAX_STEPS_PER_EPISODE, self.num_transactions),
            replace=False
        )
        self.current_step = 0
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_outcomes = []
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, info
    
    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        tx_idx = self.current_indices[self.current_step]
        true_label = self.all_labels[tx_idx]
        reward = self._calculate_reward(action, true_label)
        
        self.episode_rewards.append(reward)
        self.episode_actions.append(action)
        self.episode_outcomes.append({
            'action': RL_ACTIONS[action],
            'true_label': 'FRAUD' if true_label == 1 else 'LEGIT',
            'reward': reward,
            'fraud_score': self.all_states[tx_idx, 0]
        })
        
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= len(self.current_indices)
        
        if truncated:
            obs = self.all_states[tx_idx]
        else:
            obs = self._get_observation()
        
        info = self._get_info()
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        tx_idx = self.current_indices[self.current_step]
        return self.all_states[tx_idx].copy()
    
    def _get_info(self):
        if self.current_step < len(self.current_indices):
            tx_idx = self.current_indices[self.current_step]
            return {
                'step': self.current_step,
                'transaction_idx': int(tx_idx),
                'true_label': int(self.all_labels[tx_idx]),
                'fraud_score': float(self.all_states[tx_idx, 0]),
                'episode_reward_so_far': sum(self.episode_rewards)
            }
        else:
            return {
                'step': self.current_step,
                'episode_total_reward': sum(self.episode_rewards),
                'episode_length': len(self.episode_rewards)
            }
    
    def _calculate_reward(self, action, true_label):
        action_name = RL_ACTIONS[action]
        if true_label == 1:
            key = f"{action_name}_FRAUD"
        else:
            key = f"{action_name}_LEGIT"
        return RL_REWARDS.get(key, 0.0)
    
    def get_episode_summary(self):
        if not self.episode_outcomes:
            return {}
        
        total_reward = sum(self.episode_rewards)
        action_counts = {RL_ACTIONS[a]: self.episode_actions.count(a) for a in range(RL_NUM_ACTIONS)}
        
        fraud_caught = sum(1 for o in self.episode_outcomes
                          if o['true_label'] == 'FRAUD' and o['action'] != 'APPROVE')
        fraud_missed = sum(1 for o in self.episode_outcomes
                          if o['true_label'] == 'FRAUD' and o['action'] == 'APPROVE')
        legit_approved = sum(1 for o in self.episode_outcomes
                           if o['true_label'] == 'LEGIT' and o['action'] == 'APPROVE')
        legit_blocked = sum(1 for o in self.episode_outcomes
                          if o['true_label'] == 'LEGIT' and o['action'] != 'APPROVE')
        
        total_fraud = fraud_caught + fraud_missed
        total_legit = legit_approved + legit_blocked
        
        return {
            'total_reward': total_reward,
            'avg_reward': total_reward / len(self.episode_rewards),
            'steps': len(self.episode_rewards),
            'action_counts': action_counts,
            'fraud_caught': fraud_caught,
            'fraud_missed': fraud_missed,
            'legit_approved': legit_approved,
            'legit_blocked': legit_blocked,
            'fraud_catch_rate': fraud_caught / total_fraud if total_fraud > 0 else 0,
            'false_positive_rate': legit_blocked / total_legit if total_legit > 0 else 0
        }
