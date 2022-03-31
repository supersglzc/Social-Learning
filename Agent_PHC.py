from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd
from env import IntersectionEnv
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from utils import MovingAverage, RewardTracker, ReplayMemory

SEED = 42
IMAGE = True
GAMMA = 0.99
BATCH_SIZE = 64
MEAN_REWARD_EVERY = 300  # Episodes
LEARNING_RATE = 0.01
FRAME_STACK_SIZE = 3
N_PREDATOR = 2
N_agents = 2
env = IntersectionEnv(n_predator=N_PREDATOR, image=IMAGE)


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(nn.Conv2d(state_dim, 32, kernel_size=(5, 5), stride=(1, 1)), nn.Tanh(),
                                 nn.Dropout(0.2),
                                 nn.Flatten(),
                                 nn.Linear(32, 16), nn.ReLU(),
                                 nn.Linear(16, action_dim), )

    def forward(self, state):
        return self.net(state)

    def act(self, state, eps):
        if rd.rand() < eps:  # epsilon-greedy
            a_int = rd.randint(self.action_dim)
        else:
            # print(state)
            action = self.forward(state)[0]
            a_int = action.argmax().cpu().numpy()
        return a_int


class AgentPHC:
    def __init__(self, agent_id, player, role, punishment):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_policy = None

        self.agent_id = agent_id
        self.player = player
        self.role = role
        self.punishment = punishment

        action_dim = env.action_space.n  # 2

        state_dim = 3 * FRAME_STACK_SIZE

        self.dqn = DQN(state_dim, action_dim).to(self.device)
        self.dqn_target = deepcopy(self.dqn)

        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer_win = torch.optim.Adam(self.dqn.parameters(), lr=LEARNING_RATE / 2)
        self.optimizer_lose = torch.optim.Adam(self.dqn.parameters(), lr=LEARNING_RATE)

        self.reward_tracker = RewardTracker(MEAN_REWARD_EVERY)
        self.replay_memory = ReplayMemory(maxlen=500)

        self.learning_plot_initialised = False

    def select_action(self, state, eps):
        state = torch.tensor(state)
        state = torch.reshape(state, (9, 5, 5)).unsqueeze(0)
        action = self.dqn.act(state, eps)
        return action

    def training_step(self, batch_size=64):
        states, actions, rewards, next_states, dones = self.replay_memory.sample(batch_size)

        states = torch.tensor(states)
        states = torch.reshape(states, (batch_size, 9, 5, 5))
        next_states = torch.tensor(next_states)
        next_states = torch.reshape(next_states, (batch_size, 9, 5, 5))
        actions = torch.LongTensor(actions, device=self.device)
        rewards = torch.FloatTensor(rewards, device=self.device)
        dones = torch.FloatTensor(dones, device=self.device)

        next_q_values = self.dqn_target(next_states)
        next_q_value = next_q_values.max(1)[0]

        q_values = self.dqn(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        expected_q_value = rewards + GAMMA * next_q_value * (1 - dones)

        obj_critic = self.criterion(q_value, expected_q_value)

        if self.calculate_wol(batch_size):
            optimizer = self.optimizer_win
        else:
            optimizer = self.optimizer_lose
        optimizer.zero_grad()
        obj_critic.backward()
        optimizer.step()

        return obj_critic

    def calculate_wol(self, batch_size):
        states, actions, rewards, next_states, dones = self.replay_memory.sample(batch_size)
        # calculate win or lose
        next_states = torch.tensor(next_states)
        next_states = torch.reshape(next_states, (batch_size, 9, 5, 5))
        Q1 = np.sum(np.max(self.dqn([next_states])))
        Q2 = np.sum(np.max(self.dqn_target([next_states])))
        if Q1 > Q2:
            return 1
        else:
            return 0

    def plot_learning_curve(self, image_path=None, csv_path=None):
        colour_palette = get_cmap(name='Set1').colors
        if not self.learning_plot_initialised:
            self.fig, self.ax = plt.subplots()
            self.learning_plot_initialised = True
        self.ax.clear()

        reward_data = self.reward_tracker.get_reward_data()
        x = reward_data[:, 0]
        y = reward_data[:, 1]

        # Save raw reward data
        if csv_path:
            np.savetxt(csv_path, reward_data, delimiter=",")

        # Compute moving average
        tracker = MovingAverage(maxlen=MEAN_REWARD_EVERY)
        mean_rewards = np.zeros(len(reward_data))
        for i, (_, reward) in enumerate(reward_data):
            tracker.append(reward)
            mean_rewards[i] = tracker.mean()

        # Create plot
        self.ax.plot(x, y, alpha=0.2, c=colour_palette[0])
        self.ax.plot(x[MEAN_REWARD_EVERY // 2:], mean_rewards[MEAN_REWARD_EVERY // 2:],
                     c=colour_palette[0])
        self.ax.set_xlabel('episode')
        self.ax.set_ylabel('reward per episode')
        self.ax.grid(True, ls=':')

        # Save plot
        if image_path:
            self.fig.savefig(image_path)

    def update_target_model(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())
