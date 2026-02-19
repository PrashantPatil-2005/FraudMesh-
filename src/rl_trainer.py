import numpy as np
import json
import time
from src.dqn_agent import DQNAgent
from src.rl_baselines import RandomPolicy, RuleBasedPolicy
from src.config import (
    RL_NUM_EPISODES, RL_LOG_EVERY, RL_EVAL_EVERY,
    METRICS_DIR, RL_CHECKPOINT_PATH
)


def train_dqn(env, num_episodes=RL_NUM_EPISODES):
    """
    Train the DQN agent in the FraudResponseEnv.
    Returns the trained agent and training metrics.
    """
    print("=" * 60)
    print("PHASE 3: DQN AGENT TRAINING")
    print("=" * 60)
    
    agent = DQNAgent()
    print(f"  Device: {agent.device}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Epsilon: {agent.epsilon:.2f} -> {agent.epsilon_end:.4f}")
    print(f"  Buffer size: {agent.buffer.capacity:,}")
    print()
    
    # Tracking
    episode_rewards = []
    episode_lengths = []
    episode_epsilons = []
    episode_losses = []
    episode_fraud_catch_rates = []
    episode_false_positive_rates = []
    best_avg_reward = float('-inf')
    
    start_time = time.time()
    
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_loss_list = []
        done = False
        
        while not done:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.store_experience(obs, action, reward, next_obs, done)
            loss = agent.learn()
            if loss is not None:
                episode_loss_list.append(loss)
            
            obs = next_obs
            episode_reward += reward
        
        agent.decay_epsilon()
        
        # Track metrics
        summary = env.get_episode_summary()
        episode_rewards.append(episode_reward)
        episode_lengths.append(summary.get('steps', 0))
        episode_epsilons.append(agent.epsilon)
        avg_loss = np.mean(episode_loss_list) if episode_loss_list else 0.0
        episode_losses.append(avg_loss)
        episode_fraud_catch_rates.append(summary.get('fraud_catch_rate', 0.0))
        episode_false_positive_rates.append(summary.get('false_positive_rate', 0.0))
        
        # Logging
        if episode % RL_LOG_EVERY == 0:
            recent_rewards = episode_rewards[-RL_LOG_EVERY:]
            avg_reward = np.mean(recent_rewards)
            elapsed = time.time() - start_time
            print(f"  Episode {episode:4d}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:8.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Buffer: {len(agent.buffer):5d} | "
                  f"Time: {elapsed:.0f}s")
        
        # Save best model
        if episode >= RL_EVAL_EVERY:
            recent_avg = np.mean(episode_rewards[-RL_EVAL_EVERY:])
            if recent_avg > best_avg_reward:
                best_avg_reward = recent_avg
                agent.save()
    
    total_time = time.time() - start_time
    print(f"\n  Training complete in {total_time:.1f}s")
    print(f"  Best avg reward (window={RL_EVAL_EVERY}): {best_avg_reward:.2f}")
    print(f"  Final epsilon: {agent.epsilon:.4f}")
    print(f"  Total training steps: {agent.train_step_count:,}")
    
    # Save training metrics
    training_metrics = {
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_lengths': [int(l) for l in episode_lengths],
        'episode_epsilons': [float(e) for e in episode_epsilons],
        'episode_losses': [float(l) for l in episode_losses],
        'episode_fraud_catch_rates': [float(r) for r in episode_fraud_catch_rates],
        'episode_false_positive_rates': [float(r) for r in episode_false_positive_rates],
        'best_avg_reward': float(best_avg_reward),
        'total_time': float(total_time),
        'total_training_steps': int(agent.train_step_count)
    }
    
    metrics_path = METRICS_DIR / 'rl_training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    print(f"  Training metrics saved to {metrics_path}")
    
    return agent, training_metrics


def evaluate_baselines(env, num_episodes=50):
    """
    Evaluate Random and Rule-Based baselines for comparison.
    Returns a dict of {policy_name: metrics_dict}.
    """
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION")
    print("=" * 60)
    
    baselines = [RandomPolicy(), RuleBasedPolicy()]
    baseline_results = {}
    
    for policy in baselines:
        print(f"\n  Evaluating: {policy.name}")
        all_rewards = []
        all_summaries = []
        
        for ep in range(num_episodes):
            obs, info = env.reset()
            done = False
            ep_reward = 0.0
            
            while not done:
                action = policy.select_action(obs, training=False)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                obs = next_obs
                ep_reward += reward
            
            all_rewards.append(ep_reward)
            all_summaries.append(env.get_episode_summary())
        
        avg_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        avg_fraud_catch = np.mean([s['fraud_catch_rate'] for s in all_summaries if 'fraud_catch_rate' in s])
        avg_false_pos = np.mean([s['false_positive_rate'] for s in all_summaries if 'false_positive_rate' in s])
        
        baseline_results[policy.name] = {
            'avg_reward': float(avg_reward),
            'std_reward': float(std_reward),
            'avg_fraud_catch_rate': float(avg_fraud_catch),
            'avg_false_positive_rate': float(avg_false_pos),
            'num_episodes': num_episodes
        }
        
        print(f"    Avg Reward: {avg_reward:.2f} (+/- {std_reward:.2f})")
        print(f"    Fraud Catch Rate: {avg_fraud_catch*100:.1f}%")
        print(f"    False Positive Rate: {avg_false_pos*100:.1f}%")
    
    # Save results
    baseline_path = METRICS_DIR / 'rl_baseline_metrics.json'
    with open(baseline_path, 'w') as f:
        json.dump(baseline_results, f, indent=2)
    print(f"\n  Baseline metrics saved to {baseline_path}")
    
    return baseline_results
