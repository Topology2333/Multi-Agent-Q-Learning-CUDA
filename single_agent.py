import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
import tqdm

class GridWorld:
    def __init__(self, size, n_mines, flag_pos):
        self.size = size
        self.n_mines = n_mines
        self.flag_pos = flag_pos
        self.grid = np.zeros((size, size))
        self.mines = self.place_mines()
        self.state = (0, 0)

    def place_mines(self):
        mines = []
        while len(mines) < self.n_mines:
            pos = (np.random.randint(0, self.size), np.random.randint(0, self.size))
            if pos not in mines and pos != (0, 0) and pos != self.flag_pos:
                mines.append(pos)
                self.grid[pos] = -1
        return mines

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:  
            x = max(0, x - 1)
        elif action == 1:  
            x = min(self.size - 1, x + 1)
        elif action == 2:  
            y = max(0, y - 1)
        elif action == 3:  
            y = min(self.size - 1, y + 1)

        self.state = (x, y)

        if self.state in self.mines:
            return self.state, -10, True
        elif self.state == self.flag_pos:
            return self.state, 10, True
        else:
            return self.state, -1, False

class QLearningAgent:
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
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update(state, action, reward, next_state)
                state = next_state

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
        agent_marker, = ax.plot([], [], 'bo', markersize=8, label="Agent")

        def init():
            agent_marker.set_data([], [])
            return agent_marker,

        def update(frame):
            x, y = frame
            agent_marker.set_data(y, self.env.size - 1 - x)
            return agent_marker,

        state = self.env.reset()
        frames = [state]
        done = False

        while not done and len(frames) < max_steps:
            action = np.argmax(self.q_table[state[0], state[1]])
            state, _, done = self.env.step(action)
            frames.append(state)

        ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, repeat=False)
        ani.save(output_gif, writer=PillowWriter(fps=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--n_mines", type=int, default=40)
    parser.add_argument("--flag_x", type=int, default=31)
    parser.add_argument("--flag_y", type=int, default=31)
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--gif_path", type=str, default="")
    parser.add_argument("--max_steps", type=int, default=100)
    args = parser.parse_args()

    env = GridWorld(args.size, args.n_mines, (args.flag_x, args.flag_y))
    agent = QLearningAgent(env, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)
    agent.train(args.episodes)
    agent.visualize_policy(args.gif_path, max_steps=args.max_steps)

if __name__ == "__main__":
    main()
