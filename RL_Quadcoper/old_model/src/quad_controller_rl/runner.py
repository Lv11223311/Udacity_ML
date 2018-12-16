
from agents.ddpg import DDPG, OUNoise
from tasks.takeoff import Takeoff


class Run:
    def __init__(self, task, agent):
        self.start_time = None
        self.timestamp = None
        self. pose = None
        self.angular_velocity = None
        self.linear_acceleration  = None
        self.episode = 0
        self.task = task
        self. agent = agent

    def start(self):
        self.reset()
        self.loop()

    def loop(self):
        if self.timestamp and self.pose and self.angular_velocity \
                and self.linear_acceleration:
            cmd, done = self.task.update(self.timestamp, self.pose, self.angular_velocity, self.linear_acceleration)
            if done:
                self.reset()
            else: