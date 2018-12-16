import numpy as np
import os
import pandas as pd
from quad_controller_rl import util
from quad_controller_rl.agents.base_agent import BaseAgent
from quad_controller_rl.agents.buffer import ReplayBuffer
from quad_controller_rl.agents.actor import Actor
from quad_controller_rl.agents.critic import Critic
from quad_controller_rl.agents.noise import OUNoise
from keras.models import model_from_json


class DDPG(BaseAgent):
    """Sample agent that searches for optimal policy by DDPG.  """

    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = 3
        self.action_size = 3
        print("Original spaces:{}, {}\nConstrained spaces:{},{}".format(
            self.task.observation_space.shape, self.task.action_space.shape,
            self.state_size, self.action_size))

        self.action_low = self.task.action_space.low
        self.action_high = self.task.action_space.high

        # Actor(policy) model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        # Critic (Q-value) model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        # Noise process
        self.noise = OUNoise(self.action_size)
        # Replay Buffer
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size)
        # Policy parameters
        self.gamma = 0.99
        self.tau = 0.001

        # Save episode stats
        self.stats_filename = os.path.join(util.get_param('out'), "stats_{}.csv".format(util.get_timestamp()))
        self.stats_columns = ["episode", "total_reward"]
        self.episode_num = 1

        # Episode variables
        self.reset_episode_vars()

    def preprocess_state(self, state):
        """ Reduce state vector to relevant dimensions """
        return state[0:3]

    def preprocess_action(self, action):
        """ Return complete action vector """
        complete_action = np.zeros(self.task.action_space.shape)
        complete_action[:3] = action
        return complete_action
    
    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0

    def write_stats(self, stats):
        """ Write single episode stats to CSV file. """
        df_stats = pd.DataFrame([stats], columns=self.stats_columns)
        df_stats.to_csv(self.stats_filename, mode='a', index=False, header=not os.path.isfile(self.stats_filename))

    def step(self, state, reward, done):
        # preprocess state
        state = self.preprocess_state(state)
        # Transform state, choose action
        action = self.act(state)
        # Save experience
        if self.last_state is not None and self.last_action is not None:
            self.memory.push(self.last_state, self.last_action, reward, state, done)
            self.total_reward += reward
            self.count += 1
        # Learn from buffer
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
        # Write episode stats
        if done:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
            self.write_stats([self.episode_num, self.total_reward])
            self.episode_num += 1
            self.total_reward = 0.0

        self.last_action = action
        self.last_state = state
        # Return complete action vector 
        return self.preprocess_action(action)

    def act(self, states):
        # Choose action based on given state and policy
        states = np.reshape(states, [-1, self.state_size])
        actions = self.actor_local.model.predict(states)
        return actions + self.noise.sample()  # add some noise for exploration

    def learn(self, experiences):
        """ Update policy and value parameters using given batch of experience tuples """
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).\
            astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next_state actions and Q values from target models
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_target_next = self.critic_target.model.predict_on_batch([next_states, actions_next])
        # Compute Q targets for current states and train critic model(local)
        Q_targets = rewards + self.gamma * Q_target_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)
        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.\
                                      get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function
        # soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """ Soft update model parameters. """
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())
        # let the model more stable
        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)