from tqdm import tqdm
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

class QLearning:
    def __init__(self, env, learning_rate=0.5, gamma=0.9, min_epsilon=0.0, max_epsilon=1, decay_rate=0.003):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.decay_rate = decay_rate
        self.q_table = np.random.normal(
            size=(self.env.observation_space.n, self.env.action_space.n))

    def epsilon_greedy_policy(self, obs, epsilon):
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[obs])

    def learn(self, episodes, max_steps: int = 10000):
        rewards = []
        for episode in tqdm(range(episodes)):
            obs, _ = self.env.reset()
            epsilon = max(self.min_epsilon, self.max_epsilon -
                          self.decay_rate*episode)

            ep_reward = 0
            for _ in range(max_steps):
                action = self.epsilon_greedy_policy(obs, epsilon)
                next_obs, reward, terminated, truncated, _ = self.env.step(
                    action)
                done = np.logical_or(terminated, truncated)

                self.q_table[obs, action] = self.q_table[obs, action] + self.learning_rate * (
                    reward + (1-done)*self.gamma*np.max(self.q_table[next_obs]) - self.q_table[obs, action])
                obs = next_obs

                ep_reward += reward
                if done:
                    break

            rewards.append(ep_reward)

        return rewards


if __name__ == "__main__":
    agent = QLearning(gym.make('FrozenLake-v1', is_slippery=False))
    rewards = agent.learn(episodes=500)
    print(agent.q_table)
    plt.bar(range(len(rewards)), rewards, width=1)
    plt.pause(interval=300000)
