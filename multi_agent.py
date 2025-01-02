import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
import tqdm

class MultiAgentGridWorld:
    def __init__(self, size, n_mines, flag_pos, n_agents):
        self.size = size
        self.n_mines = n_mines
        self.flag_pos = flag_pos
        self.n_agents = n_agents
        self.grid = np.zeros((size, size))
        self.mines = self.place_mines()
        self.agent_states = [(0, 0) for _ in range(n_agents)]
        self.active_agents = [True for _ in range(n_agents)]

    def place_mines(self):
        mines = []
        while len(mines) < self.n_mines:
            pos = (np.random.randint(0, self.size), np.random.randint(0, self.size))
            if pos not in mines and pos != (0, 0) and pos != self.flag_pos:
                mines.append(pos)
                self.grid[pos] = -1
        return mines

    def reset(self):
        self.agent_states = [(0, 0) for _ in range(self.n_agents)]
        self.active_agents = [True for _ in range(self.n_agents)]
        return self.agent_states

    def step(self, agent_id, action):
        if not self.active_agents[agent_id]:
            return self.agent_states[agent_id], 0, True

        x, y = self.agent_states[agent_id]
        if action == 0: 
            x = max(0, x - 1)
        elif action == 1: 
            x = min(self.size - 1, x + 1)
        elif action == 2: 
            y = max(0, y - 1)
        elif action == 3: 
            y = min(self.size - 1, y + 1)

        self.agent_states[agent_id] = (x, y)

        if self.agent_states[agent_id] in self.mines:
            self.active_agents[agent_id] = False
            return self.agent_states[agent_id], -10, True
        elif self.agent_states[agent_id] == self.flag_pos:
            self.active_agents[agent_id] = False
            return self.agent_states[agent_id], 10, True
        else:
            return self.agent_states[agent_id], -1, False

class MultiAgentQLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((env.size, env.size, 4))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)
        x, y = state
        return np.argmax(self.q_table[x, y])

    def update(self, state, action, reward, next_state):
        x, y = state
        nx, ny = next_state
        best_next_action = np.max(self.q_table[nx, ny])
        self.q_table[x, y, action] += self.alpha * (
            reward + self.gamma * best_next_action - self.q_table[x, y, action]
        )

    def train(self, episodes):
        for _ in tqdm.tqdm(range(episodes), desc="Training"):
            states = self.env.reset()
            done_flags = [False for _ in range(self.env.n_agents)]

            while not all(done_flags):
                num_active_agents = sum(self.env.active_agents)
                if num_active_agents < 0.2 * self.env.n_agents:
                    break

                for agent_id in range(self.env.n_agents):
                    if done_flags[agent_id]:
                        continue

                    state = states[agent_id]
                    action = self.choose_action(state)
                    next_state, reward, done = self.env.step(agent_id, action)
                    self.update(state, action, reward, next_state)
                    states[agent_id] = next_state
                    done_flags[agent_id] = done

    def visualize_policy(self, output_gif, max_steps=100):
        if not output_gif:
            return
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_xlim(-0.5, self.env.size - 0.5)
        ax.set_ylim(-0.5, self.env.size - 0.5)
        ax.set_xticks(range(self.env.size))
        ax.set_yticks(range(self.env.size))
        ax.grid()

        for mine in self.env.mines:
            ax.plot(mine[1], self.env.size - 1 - mine[0], 'rx', markersize=12, label="Mine")

        ax.plot(self.env.flag_pos[1], self.env.size - 1 - self.env.flag_pos[0], 'g*', markersize=15, label="Flag")
        agent_markers = [ax.plot([], [], 'bo', markersize=8, label="Agent")[0] for _ in range(self.env.n_agents)]

        def init():
            for marker in agent_markers:
                marker.set_data([], [])
            return agent_markers

        def update(frame):
            for agent_id, pos in enumerate(frame):
                x, y = pos
                agent_markers[agent_id].set_data(y, self.env.size - 1 - x)
            return agent_markers

        states = self.env.reset()
        frames = [states]
        done_flags = [False for _ in range(self.env.n_agents)]

        while not all(done_flags) and len(frames) < max_steps:
            next_states = []
            for agent_id, state in enumerate(states):
                if done_flags[agent_id]:
                    next_states.append(state)
                    continue

                action = np.argmax(self.q_table[state[0], state[1]])
                next_state, _, done = self.env.step(agent_id, action)
                next_states.append(next_state)
                done_flags[agent_id] = done

            states = next_states
            frames.append(states)

        ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, repeat=False)
        ani.save(output_gif, writer=PillowWriter(fps=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=46)
    parser.add_argument("--n_mines", type=int, default=96)
    parser.add_argument("--flag_x", type=int, default=45)
    parser.add_argument("--flag_y", type=int, default=45)
    parser.add_argument("--n_agents", type=int, default=512)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--gif_path", type=str, default="")
    parser.add_argument("--max_steps", type=int, default=100)
    args = parser.parse_args()

    env = MultiAgentGridWorld(args.size, args.n_mines, (args.flag_x, args.flag_y), args.n_agents)
    agent = MultiAgentQLearningAgent(env, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)
    agent.train(args.episodes)
    agent.visualize_policy(args.gif_path, max_steps=args.max_steps)

if __name__ == "__main__":
    main()
