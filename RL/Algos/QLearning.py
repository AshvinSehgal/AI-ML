import numpy as np
import matplotlib.pyplot as plt

class QLearning:
    def __init__(self, env=None, rows=10, cols=10, episodes=100000, max_steps=250, 
                 max_epsilon=0.75, min_epsilon=0.1, decay_rate=0.95, alpha=0.1, gamma=0.99):
        self.env = env
        self.rows = rows
        self.cols = cols
        self.episodes = episodes
        self.max_steps = max_steps
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.alpha = alpha
        self.gamma = gamma
        self.q = np.zeros((self.env.nStates, self.env.nActions))
        self.reward_each_episode = []
        self.cum_avg_rewards = []
        self.score_each_episode = []
        self.cum_avg_score = []
        self.steps = []
        
    def learn(self):
        for e in range(self.episodes):
            self.env.reset()
            state = 0
            epsilon = self.max_epsilon
            r = 0
            step = 0
            done = False
            # if e==0:
            #     env.showEnv()
            while not done and step < self.max_steps:
                step += 1
                prob = np.random.rand()
                if prob < epsilon:
                    action = np.random.randint(4)
                else:
                    action = np.argmax(self.q[state])
                epsilon = max(self.min_epsilon, epsilon * self.decay_rate)
                nextS, reward, done = self.env.step(action)
                # if e==0:
                #     print(action)
                #     env.showEnv()
                self.q[state][action] += self.alpha * (reward + self.gamma * np.max(self.q[nextS]) - self.q[state][action])
                state = nextS
                r += reward
            self.reward_each_episode.append(r)
            self.steps.append(step)
        self.cum_avg_rewards = [np.mean(self.reward_each_episode[:i]) for i in range(1,len(self.reward_each_episode))]
            # if e < 10:
            #     env.showEnv()
            #     print('\n')
    
    def score(self):
        for e in range(self.episodes):
            self.env.reset()
            state = 0
            r = 0
            done = False
            # if e==0:
            #     env.showEnv()
            while not done:
                action = np.argmax(self.q[state])
                nextS, reward, done = self.env.step(action)
                # if e==0:
                #     print(action)
                #     env.showEnv()
                state = nextS
                r += reward
            self.score_each_episode.append(r)
        self.cum_avg_score = [np.mean(self.score_each_episode[:i]) for i in range(1,len(self.score_each_episode))]
    
    def plotRewards(self):
        plt.plot(self.cum_avg_rewards)
        plt.show()
    
    def plotScore(self):
        plt.plot(self.cum_avg_score)
        plt.show()