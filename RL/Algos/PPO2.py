import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
import gymnasium as gym
from collections import deque

class PPO:
    def __init__(self, action_space, observation_space, gamma=0.99, learning_rate=3e-4, clip_ratio=0.2, lam=0.95, entropy_coef=0.01, epochs=50):
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.observation_space = observation_space
        self.action_space = action_space

        self.actor_model = self.build_actor()
        self.critic_model = self.build_critic()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def build_actor(self):
        inputs = layers.Input(shape=self.observation_space.shape)
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        mean = layers.Dense(self.action_space.shape[0], activation='tanh')(x)
        log_std = layers.Dense(self.action_space.shape[0], activation='softplus')(x)
        return tf.keras.Model(inputs, [mean, log_std])

    def build_critic(self):
        inputs = layers.Input(shape=self.observation_space.shape)
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        value = layers.Dense(1)(x)
        return tf.keras.Model(inputs, value)

    def act(self, state):
        mean, log_std = self.actor_model(np.atleast_2d(state))
        std = tf.exp(log_std)
        dist = tfp.distributions.Normal(mean, std)
        action = dist.sample().numpy()
        return action, dist

    def compute_gae(self, rewards, values, next_values, dones):
        advantages = np.zeros_like(rewards)
        last_gae_lambda = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                delta = rewards[t] - values[t]
            else:
                delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantages[t] = last_gae_lambda = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae_lambda
        return advantages

    def update(self, states, actions, old_log_probs, returns, advantages):
        for _ in range(self.epochs):
            with tf.GradientTape() as tape:
                # Compute the loss (policy + value + entropy)
                mean, log_std = self.actor_model(np.atleast_2d(states))
                std = tf.exp(log_std)
                dist = tf.distributions.Normal(mean, std)

                new_log_probs = dist.log_prob(actions)
                ratio = tf.exp(new_log_probs - old_log_probs)
                clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

                # Value loss
                value_loss = tf.reduce_mean(tf.square(returns - self.critic_model(np.atleast_2d(states))))

                # Entropy loss for exploration
                entropy_loss = -tf.reduce_mean(dist.entropy())

                total_loss = policy_loss + value_loss - self.entropy_coef * entropy_loss

            # Compute gradients and update weights
            grads = tape.gradient(total_loss, self.actor_model.trainable_variables + self.critic_model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.actor_model.trainable_variables + self.critic_model.trainable_variables))

    def train(self, env, total_timesteps=10000, batch_size=64, update_freq=2048):
        state, _ = env.reset()
        states = []
        actions = []
        rewards = []
        ep_rewards = 0
        values = []
        old_log_probs = []
        dones = []
        next_values = []
        ep = 0

        # Collect experience
        for _ in range(total_timesteps):
            action, dist = self.act(state)
            action = action[0]
            value = self.critic_model(np.atleast_2d(state))
            log_prob = dist.log_prob(action)
            next_state, reward, done, _, _ = env.step(action)
            next_value = self.critic_model(np.atleast_2d(next_state))

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            ep_rewards += reward
            values.append(value)
            old_log_probs.append(log_prob)
            dones.append(done)
            next_values.append(next_value)

            state = next_state
            
            if done:
                print(f'Ep{ep}: {ep_rewards}')
                ep += 1
                ep_rewards= 0

            if len(states) >= update_freq:
                # Compute advantages
                advantages = self.compute_gae(rewards, values, next_values, dones)
                returns = advantages + np.array(values)

                # Update the model
                self.update(np.array(states), np.array(actions), np.array(old_log_probs), returns, advantages)
                
                # Reset memory
                states = []
                actions = []
                rewards = []
                values = []
                old_log_probs = []
                dones = []
                next_values = []

# Define and train the PPO agent
if __name__ == "__main__":
    env = gym.make('InvertedPendulum-v5')
    agent = PPO(action_space=env.action_space, observation_space=env.observation_space)
    agent.train(env)
