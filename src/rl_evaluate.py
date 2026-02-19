import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from src.config import (
    RL_ACTIONS, RL_NUM_ACTIONS, PLOTS_DIR, METRICS_DIR,
    RL_LOG_EVERY
)


def plot_training_curves(training_metrics):
    """Plot reward curves, loss, and epsilon decay."""
    print("\n  Generating training curves...")
    
    rewards = training_metrics['episode_rewards']
    losses = training_metrics['episode_losses']
    epsilons = training_metrics['episode_epsilons']
    fraud_catch = training_metrics['episode_fraud_catch_rates']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 3: DQN Training Curves', fontsize=16, fontweight='bold')
    
    # 1. Episode Rewards
    ax = axes[0, 0]
    ax.plot(rewards, alpha=0.3, color='steelblue', label='Episode Reward')
    # Rolling average
    window = min(50, len(rewards) // 5) if len(rewards) > 10 else 1
    if window > 1:
        rolling_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), rolling_avg, color='darkblue',
                linewidth=2, label=f'Rolling Avg ({window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Training Loss
    ax = axes[0, 1]
    valid_losses = [l for l in losses if l > 0]
    if valid_losses:
        ax.plot(valid_losses, color='coral', alpha=0.5)
        if len(valid_losses) > 10:
            w = min(50, len(valid_losses) // 5)
            rolling_loss = np.convolve(valid_losses, np.ones(w)/w, mode='valid')
            ax.plot(range(w-1, len(valid_losses)), rolling_loss, color='darkred', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss (Huber)')
    ax.grid(True, alpha=0.3)
    
    # 3. Epsilon Decay
    ax = axes[1, 0]
    ax.plot(epsilons, color='green', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration Rate (Epsilon)')
    ax.grid(True, alpha=0.3)
    
    # 4. Fraud Catch Rate
    ax = axes[1, 1]
    ax.plot(fraud_catch, alpha=0.3, color='purple')
    if len(fraud_catch) > 10:
        w = min(50, len(fraud_catch) // 5)
        rolling_fc = np.convolve(fraud_catch, np.ones(w)/w, mode='valid')
        ax.plot(range(w-1, len(fraud_catch)), rolling_fc, color='darkviolet', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Catch Rate')
    ax.set_title('Fraud Catch Rate')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = PLOTS_DIR / 'rl_training_curves.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def plot_action_distribution(env, agent, baselines_results, num_episodes=50):
    """Compare action distributions across DQN, Random, and Rule-Based."""
    print("  Generating action distribution comparison...")
    
    from src.rl_baselines import RandomPolicy, RuleBasedPolicy
    
    policies = {
        'DQN Agent': agent,
        'Random Policy': RandomPolicy(),
        'Rule-Based Policy': RuleBasedPolicy()
    }
    
    action_names = [RL_ACTIONS[a] for a in range(RL_NUM_ACTIONS)]
    all_distributions = {}
    
    for name, policy in policies.items():
        action_counts = np.zeros(RL_NUM_ACTIONS)
        for ep in range(num_episodes):
            obs, info = env.reset()
            done = False
            while not done:
                if hasattr(policy, 'select_action'):
                    action = policy.select_action(obs, training=False)
                else:
                    action = policy.select_action(obs, training=False)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                obs = next_obs
                action_counts[action] += 1
        
        total = action_counts.sum()
        all_distributions[name] = action_counts / total if total > 0 else action_counts
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(RL_NUM_ACTIONS)
    width = 0.25
    
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    for i, (name, dist) in enumerate(all_distributions.items()):
        bars = ax.bar(x + i * width, dist, width, label=name, color=colors[i], alpha=0.85)
        for bar, val in zip(bars, dist):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.1%}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Action', fontsize=12)
    ax.set_ylabel('Proportion', fontsize=12)
    ax.set_title('Action Distribution Comparison: DQN vs Baselines', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(action_names, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    path = PLOTS_DIR / 'rl_action_distribution.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def evaluate_dqn(env, agent, num_episodes=50):
    """Evaluate trained DQN agent and return metrics."""
    print("\n  Evaluating trained DQN agent...")
    
    all_rewards = []
    all_summaries = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        
        while not done:
            action = agent.select_action(obs, training=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            obs = next_obs
            ep_reward += reward
        
        all_rewards.append(ep_reward)
        all_summaries.append(env.get_episode_summary())
    
    dqn_metrics = {
        'avg_reward': float(np.mean(all_rewards)),
        'std_reward': float(np.std(all_rewards)),
        'min_reward': float(np.min(all_rewards)),
        'max_reward': float(np.max(all_rewards)),
        'avg_fraud_catch_rate': float(np.mean([s['fraud_catch_rate'] for s in all_summaries])),
        'avg_false_positive_rate': float(np.mean([s['false_positive_rate'] for s in all_summaries])),
        'num_episodes': num_episodes
    }
    
    print(f"    DQN Avg Reward: {dqn_metrics['avg_reward']:.2f} "
          f"(+/- {dqn_metrics['std_reward']:.2f})")
    print(f"    Fraud Catch Rate: {dqn_metrics['avg_fraud_catch_rate']*100:.1f}%")
    print(f"    False Positive Rate: {dqn_metrics['avg_false_positive_rate']*100:.1f}%")
    
    return dqn_metrics


def plot_reward_comparison(dqn_metrics, baseline_results):
    """Bar chart comparing average rewards across all policies."""
    print("  Generating reward comparison chart...")
    
    policy_names = ['DQN Agent'] + list(baseline_results.keys())
    avg_rewards = [dqn_metrics['avg_reward']] + [v['avg_reward'] for v in baseline_results.values()]
    std_rewards = [dqn_metrics['std_reward']] + [v['std_reward'] for v in baseline_results.values()]
    
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(policy_names, avg_rewards, yerr=std_rewards, capsize=5,
                  color=colors[:len(policy_names)], alpha=0.85, edgecolor='black', linewidth=0.5)
    
    for bar, val in zip(bars, avg_rewards):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Average Episode Reward', fontsize=12)
    ax.set_title('Policy Comparison: Average Reward', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    path = PLOTS_DIR / 'rl_reward_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def plot_cost_analysis(dqn_metrics, baseline_results):
    """Compare fraud catch rate vs false positive rate across policies."""
    print("  Generating cost analysis chart...")
    
    policies = {'DQN Agent': dqn_metrics}
    policies.update(baseline_results)
    
    names = list(policies.keys())
    catch_rates = [p.get('avg_fraud_catch_rate', 0) for p in policies.values()]
    fp_rates = [p.get('avg_false_positive_rate', 0) for p in policies.values()]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, [c*100 for c in catch_rates], width,
                    label='Fraud Catch Rate (%)', color='#4CAF50', alpha=0.85)
    bars2 = ax.bar(x + width/2, [f*100 for f in fp_rates], width,
                    label='False Positive Rate (%)', color='#F44336', alpha=0.85)
    
    for bar, val in zip(bars1, catch_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, fp_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Cost Analysis: Fraud Detection vs False Positives', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    path = PLOTS_DIR / 'rl_cost_analysis.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def save_combined_comparison(dqn_metrics, baseline_results, training_metrics):
    """Save a final combined comparison of all three phases."""
    print("\n  Saving combined Phase 3 comparison...")
    
    combined = {
        'phase3_rl_results': {
            'dqn_agent': dqn_metrics,
            'baselines': baseline_results,
            'training_summary': {
                'total_episodes': len(training_metrics['episode_rewards']),
                'best_avg_reward': training_metrics['best_avg_reward'],
                'total_time_seconds': training_metrics['total_time'],
                'total_training_steps': training_metrics['total_training_steps'],
                'final_epsilon': training_metrics['episode_epsilons'][-1] if training_metrics['episode_epsilons'] else None
            }
        }
    }
    
    path = METRICS_DIR / 'phase3_combined_comparison.json'
    with open(path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"    Saved: {path}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("PHASE 3 — FINAL COMPARISON")
    print("=" * 70)
    print(f"{'Policy':<20} | {'Avg Reward':>12} | {'Fraud Catch':>12} | {'False Pos':>12}")
    print("-" * 70)
    
    print(f"{'DQN Agent':<20} | {dqn_metrics['avg_reward']:>12.2f} | "
          f"{dqn_metrics['avg_fraud_catch_rate']*100:>11.1f}% | "
          f"{dqn_metrics['avg_false_positive_rate']*100:>11.1f}%")
    
    for name, metrics in baseline_results.items():
        print(f"{name:<20} | {metrics['avg_reward']:>12.2f} | "
              f"{metrics['avg_fraud_catch_rate']*100:>11.1f}% | "
              f"{metrics['avg_false_positive_rate']*100:>11.1f}%")
    
    print("=" * 70)
    
    return combined
