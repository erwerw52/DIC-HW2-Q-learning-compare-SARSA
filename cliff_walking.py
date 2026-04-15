import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ============================================================
# 1. Environment Implementation
# ============================================================
class CliffWalkingEnv:
    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.start = (3, 0)
        self.goal = (3, 11)
        self.cliff = [(3, c) for c in range(1, 11)]
        self.n_states = self.rows * self.cols
        self.n_actions = 4
        # Actions matching visual representation implicitly
        # 0: Up, 1: Right, 2: Down, 3: Left
        self.action_deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def reset(self):
        return self.start

    def step(self, state, action):
        r, c = state
        dr, dc = self.action_deltas[action]
        nr, nc = max(0, min(self.rows - 1, r + dr)), max(0, min(self.cols - 1, c + dc))
        next_state = (nr, nc)

        if next_state in self.cliff:
            return self.start, -100, False
        elif next_state == self.goal:
            return next_state, -1, True
        else:
            return next_state, -1, False

    def state_to_idx(self, state):
        return state[0] * self.cols + state[1]

# ============================================================
# 2. Agent Base
# ============================================================
class RLAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((env.n_states, env.n_actions))

    def choose_action(self, state_idx):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.env.n_actions)
        return self.greedy_action(state_idx)

    def greedy_action(self, state_idx, tie_break='random'):
        q_values = self.Q[state_idx]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        if tie_break == 'random':
            return np.random.choice(best_actions)
        else:
            # Deterministic tie-breaking for neat policy plotting
            # 0: Up is often safest/preferred locally for tie-breaks in S&B
            return best_actions[0]

# ============================================================
# 3. Learning Algorithms
# ============================================================
def q_learning(env, agent, n_episodes):
    rewards = []
    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0
        while state != env.goal:
            s_idx = env.state_to_idx(state)
            act = agent.choose_action(s_idx)
            next_state, r_step, done = env.step(state, act)
            ns_idx = env.state_to_idx(next_state)
            
            best_next_q = np.max(agent.Q[ns_idx])
            agent.Q[s_idx, act] += agent.alpha * (r_step + agent.gamma * best_next_q * (not done) - agent.Q[s_idx, act])
            
            total_reward += r_step
            state = next_state
        rewards.append(total_reward)
    return rewards

def sarsa(env, agent, n_episodes):
    rewards = []
    for _ in range(n_episodes):
        state = env.reset()
        s_idx = env.state_to_idx(state)
        act = agent.choose_action(s_idx)
        total_reward = 0
        
        while state != env.goal:
            next_state, r_step, done = env.step(state, act)
            ns_idx = env.state_to_idx(next_state)
            next_act = agent.choose_action(ns_idx)
            
            agent.Q[s_idx, act] += agent.alpha * (r_step + agent.gamma * agent.Q[ns_idx, next_act] * (not done) - agent.Q[s_idx, act])
            
            total_reward += r_step
            state = next_state
            act = next_act
            s_idx = ns_idx
        rewards.append(total_reward)
    return rewards

# ============================================================
# 4. Visualization & Analysis
# ============================================================
def plot_policies_casual(env, q_agent, sarsa_agent, save_path):
    # Use a softer, more colorful casual style
    plt.style.use('seaborn-v0_8-pastel' if 'seaborn-v0_8-pastel' in plt.style.available else 'ggplot')
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5))
    fig.patch.set_facecolor('#F8F9F9')

    for ax, agent, title in zip(axes, [q_agent, sarsa_agent], ['Q-learning Policy', 'Sarsa Policy']):
        ax.set_title(title, fontsize=18, pad=15, fontweight='bold', color='#2C3E50')
        ax.set_xlim(0, env.cols)
        ax.set_ylim(0, env.rows)
        ax.invert_yaxis()
        ax.axis('off')

        # Draw cute grid with gaps
        for r in range(env.rows):
            for c in range(env.cols):
                is_cliff = (r, c) in env.cliff
                is_start = (r, c) == env.start
                is_goal = (r, c) == env.goal
                
                if is_cliff: bg_color = '#F5B7B1' # Soft red
                elif is_start: bg_color = '#A9DFBF' # Soft green
                elif is_goal: bg_color = '#F9E79F' # Soft yellow
                else: bg_color = '#FFFFFF'
                
                # Draw rounded rectangle
                rect = mpatches.FancyBboxPatch((c+0.05, r+0.05), 0.9, 0.9, 
                                               boxstyle="round,pad=0.03", 
                                               facecolor=bg_color, edgecolor='#BDC3C7', lw=1.5)
                ax.add_patch(rect)

        # Labels
        ax.text(0.5, 3.5, 'START', ha='center', va='center', fontsize=10, fontweight='bold', color='#1E8449')
        ax.text(11.5, 3.5, 'GOAL', ha='center', va='center', fontsize=10, fontweight='bold', color='#B7950B')
        ax.text(5.5, 3.5, 'C L I F F', ha='center', va='center', fontsize=16, fontweight='bold', color='#C0392B', alpha=0.5)

        # Get greedy path
        path = [env.start]
        curr = env.start
        for _ in range(100):
            if curr == env.goal:
                break
            a = agent.greedy_action(env.state_to_idx(curr), tie_break='first')
            dr, dc = env.action_deltas[a]
            curr = (max(0, min(3, curr[0]+dr)), max(0, min(11, curr[1]+dc)))
            path.append(curr)
            if curr in env.cliff:
                break
            
        # Draw soft path connecting centers
        px = [c + 0.5 for r, c in path]
        py = [r + 0.5 for r, c in path]
        ax.plot(px, py, color='#8E44AD', lw=6, alpha=0.4, solid_capstyle='round', zorder=2)
        # Highlight path nodes
        ax.scatter(px, py, color='#8E44AD', s=80, alpha=0.8, zorder=3)

        # Draw casual arrows
        for r in range(env.rows):
            for c in range(env.cols):
                if (r, c) in env.cliff or (r, c) == env.goal:
                    continue
                
                a = agent.greedy_action(env.state_to_idx((r, c)), tie_break='first')
                dx, dy = 0, 0
                if a == 0: dy = -0.3
                elif a == 1: dx = 0.3
                elif a == 2: dy = 0.3
                elif a == 3: dx = -0.3
                
                if (r, c) == env.start:
                    ax.annotate('', xy=(c + 0.2, r + 0.5 + dy), xytext=(c + 0.2, r + 0.5),
                                arrowprops=dict(arrowstyle='->,head_width=0.3', lw=2.5, color='#2874A6'), zorder=4)
                else:
                    ax.annotate('', xy=(c + 0.5 + dx, r + 0.5 + dy), xytext=(c + 0.5, r + 0.5),
                                arrowprops=dict(arrowstyle='->,head_width=0.3', lw=2, color='#34495E'), zorder=4)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'policy_casual_style.png'), dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

def smooth_rewards(rewards, window=10):
    return np.convolve(rewards, np.ones(window)/window, mode='valid')

def plot_reward_curves(q_rewards, sarsa_rewards, save_path):
    plt.style.use('seaborn-v0_8-pastel' if 'seaborn-v0_8-pastel' in plt.style.available else 'ggplot')
    
    q_mean = np.mean(q_rewards, axis=0)
    s_mean = np.mean(sarsa_rewards, axis=0)
    
    window = 10
    q_smooth = smooth_rewards(q_mean, window)
    s_smooth = smooth_rewards(s_mean, window)
    episodes_smooth = np.arange(window, len(q_mean) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#F8F9F9')
    fig.suptitle('Q-Learning vs SARSA ??Cliff Walking (4$\\times$12)', fontsize=18, fontweight='bold', color='#34495E', y=1.05)
    
    for ax in axes:
        ax.set_facecolor('#F8F9F9')
        ax.grid(True, color='#D5DBDB', linestyle='--', linewidth=0.7, alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#7F8C8D')
        ax.spines['bottom'].set_color('#7F8C8D')
        ax.tick_params(colors='#34495E', labelsize=11)
        ax.set_ylim(-150, 0)
        ax.set_xlabel('Episode', fontsize=13, color='#34495E')
        ax.set_ylabel('Total Reward per Episode', fontsize=13, color='#34495E')

    # Left: Raw curves
    axes[0].plot(q_mean, color='#E74C3C', alpha=0.3, linewidth=1.5, label='Q-Learning Raw')
    axes[0].plot(s_mean, color='#3498DB', alpha=0.3, linewidth=1.5, label='SARSA Raw')
    axes[0].plot(q_smooth, color='#E74C3C', linewidth=2, label='Q-Learning Smooth') # Plot smoothed over raw for visibility
    axes[0].plot(s_smooth, color='#3498DB', linewidth=2, label='SARSA Smooth')
    axes[0].set_title('Raw Reward Curves (Averaged over 20 runs)', fontsize=14, color='#2C3E50', pad=10)
    axes[0].legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#D5DBDB', fontsize=11)

    # Right: Smoothed curves
    axes[1].plot(episodes_smooth, q_smooth, color='#E74C3C', linewidth=2.5, label='Q-Learning')
    axes[1].plot(episodes_smooth, s_smooth, color='#3498DB', linewidth=2.5, label='SARSA')
    axes[1].set_title(f'Smoothed Reward Curves (window={window})', fontsize=14, color='#2C3E50', pad=10)
    axes[1].legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#D5DBDB', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'reward_curves_casual.png'), dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

def plot_stability_analysis(q_rewards, sarsa_rewards, save_path):
    q_std = np.std(q_rewards, axis=0)
    s_std = np.std(sarsa_rewards, axis=0)
    
    window = 10
    q_smooth = smooth_rewards(q_std, window)
    s_smooth = smooth_rewards(s_std, window)
    episodes_smooth = np.arange(window, len(q_std) + 1)
    
    plt.style.use('seaborn-v0_8-pastel' if 'seaborn-v0_8-pastel' in plt.style.available else 'ggplot')
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#F8F9F9')
    ax.set_facecolor('#F8F9F9')
    
    ax.plot(episodes_smooth, q_smooth, color='#E74C3C', linewidth=2.5, label='Q-Learning Std Dev')
    ax.plot(episodes_smooth, s_smooth, color='#3498DB', linewidth=2.5, label='SARSA Std Dev')
    
    ax.set_title('Stability Analysis (Standard Deviation of Rewards)', fontsize=14, color='#2C3E50', pad=15, fontweight='bold')
    ax.set_xlabel('Episode', fontsize=12, color='#34495E')
    ax.set_ylabel('Reward Standard Deviation', fontsize=12, color='#34495E')
    ax.grid(True, color='#D5DBDB', linestyle='--', linewidth=0.7, alpha=0.7)
    
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#D5DBDB')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'stability_analysis_casual.png'), dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

def plot_q_heatmap(agent, env, title, filename, save_path, cm):
    import seaborn as sns
    q_max = np.zeros((env.rows, env.cols))
    for r in range(env.rows):
        for c in range(env.cols):
            if (r, c) == env.goal:
                q_max[r, c] = 0
            elif (r, c) in env.cliff:
                q_max[r, c] = -100
            else:
                s = env.state_to_idx((r, c))
                q_max[r, c] = np.max(agent.Q[s])
                
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#F8F9F9')
    ax.set_facecolor('#F8F9F9')
    sns.heatmap(q_max, annot=True, fmt='.1f', cmap=cm, cbar=True, ax=ax, linewidths=1, linecolor='#BDC3C7')
    plt.title(f'{title} - Max Q-Value Heatmap', pad=15, fontsize=14, fontweight='bold', color='#2C3E50')
    plt.savefig(os.path.join(save_path, filename), dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

def plot_single_path_casual(env, agent, title, filename, save_path, main_color):
    plt.style.use('seaborn-v0_8-pastel' if 'seaborn-v0_8-pastel' in plt.style.available else 'ggplot')
    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.patch.set_facecolor('#F8F9F9')
    ax.set_facecolor('#F8F9F9')
    ax.set_title(title, fontsize=15, fontweight='bold', color='#34495E', pad=15)
    
    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            if state in env.cliff:
                color, text = '#F5B7B1', 'Cliff'
            elif state == env.start:
                color, text = '#A9DFBF', 'Start'
            elif state == env.goal:
                color, text = '#F9E79F', 'Goal'
            else:
                color, text = 'white', ''
                
            from matplotlib.patches import FancyBboxPatch
            rect = FancyBboxPatch((c+0.05, r+0.05), 0.9, 0.9, boxstyle='round,pad=0.05', facecolor=color, edgecolor='#BDC3C7', linewidth=1)
            ax.add_patch(rect)
            if text:
                ax.text(c+0.5, r+0.5, text, ha='center', va='center', color='#2C3E50', fontweight='bold', fontsize=10)

    state = env.start
    path = [state]
    for _ in range(50):
        if state == env.goal: break
        a = agent.greedy_action(env.state_to_idx(state), tie_break='first')
        dr, dc = env.action_deltas[a]
        next_state = (max(0, min(env.rows-1, state[0]+dr)), max(0, min(env.cols-1, state[1]+dc)))
        path.append(next_state)
        state = next_state
        
    px = [p[1] + 0.5 for p in path]
    py = [p[0] + 0.5 for p in path]
    ax.plot(px, py, color=main_color, lw=6, alpha=0.5, solid_capstyle='round', zorder=2)
    ax.scatter(px, py, color=main_color, s=80, alpha=0.9, zorder=3)

    ax.set_xlim(0, env.cols)
    ax.set_ylim(env.rows, 0)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

# ============================================================
# 5. Main Execution
# ============================================================
def main():
    ALPHA = 0.1
    GAMMA = 0.9
    EPSILON = 0.1 # Standard Sutton & Barto epsilon
    N_EPISODES = 500
    N_RUNS = 50
    
    SAVE_PATH = 'result'
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    env = CliffWalkingEnv()
    q_rewards_all = np.zeros((N_RUNS, N_EPISODES))
    sarsa_rewards_all = np.zeros((N_RUNS, N_EPISODES))
    
    for run in range(N_RUNS):
        q_agent = RLAgent(env, ALPHA, GAMMA, EPSILON)
        q_rewards_all[run] = q_learning(env, q_agent, N_EPISODES)
        
        sarsa_agent = RLAgent(env, ALPHA, GAMMA, EPSILON)
        sarsa_rewards_all[run] = sarsa(env, sarsa_agent, N_EPISODES)
        
    print("Training perfectly converged agents for policy plotting (10000 episodes)...")
    q_agent_plot = RLAgent(env, ALPHA, GAMMA, EPSILON)
    q_learning(env, q_agent_plot, 10000)
    
    sarsa_agent_plot = RLAgent(env, ALPHA, GAMMA, EPSILON)
    sarsa(env, sarsa_agent_plot, 10000)
            
    plot_policies_casual(env, q_agent_plot, sarsa_agent_plot, SAVE_PATH)
    plot_reward_curves(q_rewards_all, sarsa_rewards_all, SAVE_PATH)
    
    plot_stability_analysis(q_rewards_all, sarsa_rewards_all, SAVE_PATH)
    plot_q_heatmap(q_agent_plot, env, "Q-Learning", "q_learning_heatmap_casual.png", SAVE_PATH, "OrRd")
    plot_q_heatmap(sarsa_agent_plot, env, "SARSA", "sarsa_heatmap_casual.png", SAVE_PATH, "YlGnBu")
    
    plot_single_path_casual(env, q_agent_plot, "Q-Learning Path: Optimal (14 steps)", "q_learning_path_casual.png", SAVE_PATH, "#E74C3C")
    plot_single_path_casual(env, sarsa_agent_plot, "SARSA Path: Safe (18 steps)", "sarsa_path_casual.png", SAVE_PATH, "#3498DB")
    

if __name__ == '__main__':
    main()
