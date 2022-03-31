from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd
from env import IntersectionEnv
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from utils import MovingAverage, RewardTracker, ReplayMemory, ReservoirBuffer

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
            action = self.forward(state)[0]
            a_int = action.argmax().cpu().numpy()
        return a_int


class Policy(DQN):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)
        self.net = nn.Sequential(nn.Conv2d(state_dim, 32, kernel_size=(5, 5), stride=(1, 1)), nn.Tanh(),
                                 nn.Dropout(0.2),
                                 nn.Flatten(),
                                 nn.Linear(32, 16), nn.Tanh(),
                                 nn.Linear(16, action_dim), nn.Softmax(dim=1))

    def forward(self, state):
        return self.net(state)

    def act(self, state, eps):
        with torch.no_grad():
            distribution = self.forward(state)
            action = distribution.multinomial(1).item()
        return action


class AgentFP:
    def __init__(self, agent_id, player, role, punishment):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_policy = None

        self.agent_id = agent_id
        self.role = role
        self.player = player
        self.punishment = punishment

        action_dim = env.action_space.n  # 2

        state_dim = 3 * FRAME_STACK_SIZE

        self.dqn = DQN(state_dim, action_dim).to(self.device)
        self.dqn_target = deepcopy(self.dqn)
        self.policy = Policy(state_dim, action_dim).to(self.device)

        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer_dqn = torch.optim.Adam(self.dqn.parameters(), lr=LEARNING_RATE)
        self.optimizer_policy = torch.optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)

        self.reward_tracker = RewardTracker(MEAN_REWARD_EVERY)
        self.replay_memory = ReplayMemory(maxlen=500)
        self.reservoir_buffer = ReservoirBuffer(500)
        self.learning_plot_initialised = False

    def select_action(self, state, eps, is_best_response):
        state = torch.tensor(state)
        state = torch.reshape(state, (9, 5, 5)).unsqueeze(0)
        if is_best_response:
            action = self.dqn.act(state, eps)
        else:
            action = self.policy.act(state, eps)
        return action

    def training_step(self, batch_size=64):
        # update_target_model DQN --------------------------------------------------------------------------------------
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

        self.optimizer_dqn.zero_grad()
        obj_critic.backward()
        self.optimizer_dqn.step()

        # update_target_model policy -----------------------------------------------------------------------------------
        states2, actions2 = self.reservoir_buffer.sample(batch_size)

        states2 = torch.tensor(states2)
        states2 = torch.reshape(states2, (batch_size, 9, 5, 5))
        actions2 = torch.LongTensor(actions2, device=self.device)

        probs = self.policy(states2)
        probs_with_actions = probs.gather(1, actions2.unsqueeze(1))
        log_probs = probs_with_actions.log()

        obj_critic2 = -1 * log_probs.mean()

        self.optimizer_policy.zero_grad()
        obj_critic2.backward()
        self.optimizer_policy.step()

        return obj_critic, obj_critic2

    def plot_learning_curve(self, image_path=None, csv_path=None):
        """
        Plot the rewards per episode collected during training
        """

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
