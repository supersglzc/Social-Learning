import numpy as np
import itertools
import random
import numpy.random as rd
import math
from collections import deque


class ReplayMemory(deque):
    def sample(self, batch_size):
        indices = np.random.randint(len(self), size=batch_size)
        batch = [self[index] for index in indices]
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)]
        return states, actions, rewards, next_states, dones


class RewardTracker:
    def __init__(self, max_len):
        self.moving_average = deque([-np.inf for _ in range(max_len)], maxlen=max_len)
        self.max_len = max_len
        self.episode_rewards = []

    def __repr__(self):
        return self.moving_average.__repr__()

    def append(self, reward):
        self.moving_average.append(reward)
        self.episode_rewards.append(reward)

    def mean(self):
        return sum(self.moving_average) / self.max_len

    def get_reward_data(self):
        episodes = np.array(
            [i for i in range(len(self.episode_rewards))]).reshape(-1, 1)

        rewards = np.array(self.episode_rewards).reshape(-1, 1)
        return np.concatenate((episodes, rewards), axis=1)


class MovingAverage(deque):
    def mean(self):
        return sum(self) / len(self)


class ReservoirBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action):
        state = np.expand_dims(state, 0)
        self.buffer.append((state, action))

    def sample(self, batch_size):
        n = len(self.buffer)
        reservoir = list(itertools.islice(self.buffer, 0, batch_size))
        threshold = batch_size * 4
        idx = batch_size
        while idx < n and idx <= threshold:
            m = rd.randint(0, idx)
            if m < batch_size:
                reservoir[m] = self.buffer[idx]
            idx += 1

        while idx < n:
            p = float(batch_size) / idx
            u = rd.random()
            g = math.floor(math.log(u) / math.log(1 - p))
            idx = idx + g
            if idx < n:
                k = rd.randint(0, batch_size - 1)
                reservoir[k] = self.buffer[idx]
            idx += 1

        state, action = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), np.asarray(action)

    def __len__(self):
        return len(self.buffer)
