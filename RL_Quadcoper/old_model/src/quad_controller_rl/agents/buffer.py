from collections import namedtuple
import random

Experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """ Fixed-size circular buffer to store experience tuples """

    def __init__(self, size = 10000):
        self.size = size
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Save a transition."""
        e = Experience(state, action, reward, next_state, done)
        if len(self.memory) < self.size:
            self.memory.append(e)
        self.memory[self.position] = e
        self.position = (self.position + 1) % self.size

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
