import numpy as np
import matplotlib.pyplot as plt
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
        # Actions: 0: Up, 1: Right, 2: Down, 3: Left
        self.action_deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.current_state = self.start

    def reset(self):
        self.current_state = self.start
        return self.current_state

    def step(self, action):
        r, c = self.current_state
        dr, dc = self.action_deltas[action]
        next_r = max(0, min(self.rows - 1, r + dr))
        next_c = max(0, min(self.cols - 1, c + dc))
        next_state = (next_r, next_c)

        if next_state in self.cliff:
            reward = -100
            next_state = self.start
            done = False  # The episode continues, but agent is reset
        elif next_state == self.goal:
            reward = -1
            done = True
        else:
            reward = -1
            done = False

        self.current_state = next_state
        return next_state, reward, done

    def state_to_idx(self, state):
        return state[0] * self.cols + state[1]

    def idx_to_state(self, idx):
        return (idx // self.cols, idx % self.cols)

# ============================================================
# 2. Agent & Policy
# ============================================================
def epsilon_greedy(Q, state_idx, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.randint(4)
    else:
        q_values = Q[state_idx]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return np.random.choice(best_actions)

def q_learning_episode(env, Q, alpha, gamma, epsilon):
    state = env.reset()
    state_idx = env.state_to_idx(state)
    total_reward = 0
    done = False
    
    while not done:
        action = epsilon_greedy(Q, state_idx, epsilon)
        next_state, reward, done = env.step(action)
        next_state_idx = env.state_to_idx(next_state)
        
        total_reward += reward
        
        best_next_action = np.argmax(Q[next_state_idx])
        td_target = reward + gamma * Q[next_state_idx, best_next_action] * (not done)
        td_error = td_target - Q[state_idx, action]
        
        Q[state_idx, action] += alpha * td_error
        state_idx = next_state_idx
        
    return total_reward

def sarsa_episode(env, Q, alpha, gamma, epsilon):
    state = env.reset()
    state_idx = env.state_to_idx(state)
    action = epsilon_greedy(Q, state_idx, epsilon)
    total_reward = 0
    done = False
    
    while not done:
        next_state, reward, done = env.step(action)
        next_state_idx = env.state_to_idx(next_state)
        next_action = epsilon_greedy(Q, next_state_idx, epsilon)
        
        total_reward += reward
        
        td_target = reward + gamma * Q[next_state_idx, next_action] * (not done)
        td_error = td_target - Q[state_idx, action]
        
        Q[state_idx, action] += alpha * td_error
        state_idx = next_state_idx
        action = next_action
        
    return total_reward

# ============================================================
# 3. Training Loop & Evaluation
# ============================================================
def get_greedy_path(env, Q):
    state = env.start
    path = [state]
    while state != env.goal:
        s_idx = env.state_to_idx(state)
        action = np.argmax(Q[s_idx])
        dr, dc = env.action_deltas[action]
        r, c = state
        next_r = max(0, min(env.rows - 1, r + dr))
        next_c = max(0, min(env.cols - 1, c + dc))
        state = (next_r, next_c)
        if state in env.cliff:
            path.append(state)
            break # fell off
        path.append(state)
        if len(path) > 100: # loop prevention
            break
    return path

def plot_paths(env, path_q, path_sarsa, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    for ax, path, title in zip(axes, [path_q, path_sarsa], ['Q-Learning Path', 'SARSA Path']):
        # Draw grid
        for i in range(env.rows + 1):
            ax.axhline(i, color='black', lw=0.5)
        for j in range(env.cols + 1):
            ax.axvline(j, color='black', lw=0.5)
            
        # Draw cliff
        for r, c in env.cliff:
            rect = plt.Rectangle((c, env.rows - 1 - r), 1, 1, facecolor='gray')
            ax.add_patch(rect)
            
        # Draw Start and Goal
        ax.text(env.start[1] + 0.5, env.rows - 1 - env.start[0] + 0.5, 'S', ha='center', va='center', fontsize=20, color='green')
        ax.text(env.goal[1] + 0.5, env.rows - 1 - env.goal[0] + 0.5, 'G', ha='center', va='center', fontsize=20, color='gold')
        
        # Plot path
        xs = [c + 0.5 for r, c in path]
        ys = [env.rows - 1 - r + 0.5 for r, c in path]
        ax.plot(xs, ys, color='blue', lw=3, marker='o', markersize=5)
        
        ax.set_title(title)
        ax.set_xlim(0, env.cols)
        ax.set_ylim(0, env.rows)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'paths.png'))
    plt.close()

def main():
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    episodes = 500
    runs = 50
    
    env = CliffWalkingEnv()
    
    rewards_q_learning = np.zeros((runs, episodes))
    rewards_sarsa = np.zeros((runs, episodes))
    
    print("Training Q-Learning and SARSA...")
    
    for run in range(runs):
        Q_q_learning = np.zeros((env.n_states, env.n_actions))
        Q_sarsa = np.zeros((env.n_states, env.n_actions))
        
        for ep in range(episodes):
            rewards_q_learning[run, ep] = q_learning_episode(env, Q_q_learning, alpha, gamma, epsilon)
            rewards_sarsa[run, ep] = sarsa_episode(env, Q_sarsa, alpha, gamma, epsilon)
            
    print("Training complete.")
    
    # Averaged rewards
    avg_rewards_q = np.mean(rewards_q_learning, axis=0)
    avg_rewards_sarsa = np.mean(rewards_sarsa, axis=0)
    
    # Create result directory
    os.makedirs('result', exist_ok=True)
    
    # 4.1 Plot Reward Curves
    plt.figure(figsize=(10, 6))
    plt.plot(avg_rewards_q, label='Q-Learning', alpha=0.8)
    plt.plot(avg_rewards_sarsa, label='SARSA', alpha=0.8)
    plt.ylim(-150, 0)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward during episode')
    plt.title('Q-Learning vs SARSA on Cliff Walking (Averaged over 50 runs)')
    plt.legend()
    plt.grid(True)
    plt.savefig('result/reward_curves.png')
    plt.close()
    
    # 4.2 Map Visualization
    path_q = get_greedy_path(env, Q_q_learning)
    path_sarsa = get_greedy_path(env, Q_sarsa)
    plot_paths(env, path_q, path_sarsa, 'result')
    
    print("Plots saved in 'result/' directory.")

if __name__ == '__main__':
    main()
