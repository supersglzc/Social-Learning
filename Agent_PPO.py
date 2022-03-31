import torch
import torch.nn as nn
import numpy as np
from collections import deque
import numpy.random as rd
from env import IntersectionEnv
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from utils import MovingAverage, RewardTracker

SEED = 42
IMAGE = True
GAMMA = 0.99
BATCH_SIZE = 64
MEAN_REWARD_EVERY = 300  # Episodes
FRAME_STACK_SIZE = 3
N_PREDATOR = 2
N_agents = 2
env = IntersectionEnv(n_predator=N_PREDATOR, image=IMAGE)


class PPOReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, if_discrete):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.action_dim = 1 if if_discrete else action_dim

        other_dim = 1 + 1 + self.action_dim + action_dim
        self.buf_other = np.empty((max_len, other_dim), dtype=np.float32)
        self.buf_state = np.empty((max_len, 5, 5, 9), dtype=np.float32)

    def append_buffer(self, state, other):
        self.buf_state[self.next_idx] = state
        self.buf_other[self.next_idx] = other
        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

    def extend_buffer(self, state, other):  # CPU array to CPU array
        size = len(other)
        next_idx = self.next_idx + size
        if next_idx > self.max_len:
            if next_idx > self.max_len:
                self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
                self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
            self.if_full = True
            next_idx = next_idx - self.max_len

            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other
        self.next_idx = next_idx

    def extend_buffer_from_list(self, trajectory_list):
        state_ary = np.array([item[0] for item in trajectory_list], dtype=np.float32)
        other_ary = np.array([item[1] for item in trajectory_list], dtype=np.float32)
        self.extend_buffer(state_ary, other_ary)

    def sample_batch(self, batch_size):
        indices = rd.randint(self.now_len - 1, size=batch_size)
        r_m_a = self.buf_other[indices]
        return (r_m_a[:, 0:1],  # reward
                r_m_a[:, 1:2],  # mask = 0.0 if done else gamma
                r_m_a[:, 2:],  # action
                self.buf_state[indices],  # state
                self.buf_state[indices + 1])  # next_state

    def sample_all(self):
        all_other = torch.as_tensor(self.buf_other[:self.now_len], device=self.device)
        return (all_other[:, 0],  # reward
                all_other[:, 1],  # mask = 0.0 if done else gamma
                all_other[:, 2:2 + self.action_dim],  # action
                all_other[:, 2 + self.action_dim:],  # noise
                self.buf_state[:self.now_len])  # state

    def update_now_len(self):
        self.now_len = self.max_len if self.if_full else self.next_idx

    def empty_buffer(self):
        self.next_idx = 0
        self.now_len = 0
        self.if_full = False


class ActorDiscretePPO(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(state_dim, 32, kernel_size=(5, 5), stride=(1, 1)), nn.Tanh(),
                                 nn.Dropout(0.2),
                                 nn.Flatten(),
                                 nn.Linear(32, 16), nn.Tanh(),
                                 nn.Linear(16, action_dim), )
        layer_norm(self.net[-1], std=0.1)  # output layer for action

        self.a_logstd = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        self.soft_max = nn.Softmax(dim=1)
        self.Categorical = torch.distributions.Categorical

    def forward(self, state):
        return self.net(state)

    def get_action_prob(self, state):
        action_prob = self.soft_max(self.net(state))
        action_int = torch.multinomial(action_prob, 1, True)
        return action_int.squeeze(1), action_prob

    def get_new_logprob_entropy(self, state, action):
        a_prob = self.soft_max(self.net(state))
        dist = self.Categorical(a_prob)
        a_int = action.squeeze(1).long()
        return dist.log_prob(a_int), dist.entropy().mean()

    def get_old_logprob(self, action, a_prob):
        dist = self.Categorical(a_prob)
        return dist.log_prob(action.long().squeeze(1))


class CriticAdv(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(state_dim, 32, kernel_size=(5, 5), stride=(1, 1)), nn.Tanh(),
                                 nn.Dropout(0.2),
                                 nn.Flatten(),
                                 nn.Linear(32, 16), nn.Tanh(),
                                 nn.Linear(16, 1), )
        layer_norm(self.net[-1], std=0.5)  # output layer for Q value

    def forward(self, state):
        return self.net(state)  # Q value


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


class AgentPPO:
    def __init__(self, agent_id, player, role, punishment, if_per_or_gae=False):
        super().__init__()
        self.ratio_clip = 0.20  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.02
        self.lambda_gae_adv = 0.98
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_reward = self.compute_reward_gae if if_per_or_gae else self.compute_reward_raw

        self.agent_id = agent_id
        self.role = role
        self.player = player
        self.punishment = punishment

        action_dim = env.action_space.n  # 2
        state_dim = 3 * FRAME_STACK_SIZE
        self.act = ActorDiscretePPO(state_dim, action_dim).to(self.device)
        self.cri = CriticAdv(state_dim).to(self.device)

        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': 0.001},
                                           {'params': self.cri.parameters(), 'lr': 0.001}])
        self.reward_tracker = RewardTracker(MEAN_REWARD_EVERY)
        self.replay_buffer = PPOReplayBuffer(max_len=500, state_dim=9, action_dim=action_dim,
                                             if_discrete=True)
        self.learning_plot_initialised = False

    def select_action(self, state) -> tuple:
        state = torch.tensor(state)
        state = torch.reshape(state, (9, 5, 5)).unsqueeze(0)
        actions, action_prob = self.act.get_action_prob(state)
        return actions[0].detach().cpu().numpy(), action_prob[0].detach().cpu().numpy()

    def update_net(self, batch_size=64, repeat_times=1):
        self.replay_buffer.update_now_len()
        buf_len = self.replay_buffer.now_len
        '''compute reverse reward'''
        with torch.no_grad():
            buf_reward, buf_mask, buf_action, buf_a_prob, buf_state = self.replay_buffer.sample_all()

            buf_state = torch.tensor(buf_state)
            buf_state = torch.reshape(buf_state, (buf_len, 9, 5, 5))

            bs = 2 ** 3
            buf_value = torch.cat([self.cri(buf_state[i:i + bs]) for i in range(0, buf_state.shape[0], bs)], dim=0)

            buf_logprob = self.act.get_old_logprob(buf_action, buf_a_prob)

            buf_r_sum, buf_advantage = self.compute_reward(self, buf_len, buf_reward, buf_mask, buf_value)
            del buf_reward, buf_mask, buf_a_prob

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = obj_actor = new_logprob = None
        for update_c in range(int(buf_len / batch_size * repeat_times)):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            r_sum = buf_r_sum[indices]
            logprob = buf_logprob[indices]
            advantage = buf_advantage[indices]

            new_logprob, obj_entropy = self.act.get_new_logprob_entropy(state, action)
            ratio = (new_logprob - logprob.detach()).exp()
            obj_surrogate1 = advantage * ratio
            obj_surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            value = self.cri(state).squeeze(1)
            obj_critic = self.criterion(value, r_sum)

            obj_united = obj_actor + obj_critic / (r_sum.std() + 1e-6)
            self.optimizer.zero_grad()
            obj_united.backward()
            self.optimizer.step()

        logging_tuple = (obj_critic.item(), obj_actor.item(), new_logprob.mean().item())
        return logging_tuple

    @staticmethod
    def compute_reward_raw(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum
        pre_r_sum = 0  # reward sum of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_advantage = buf_r_sum - (buf_mask * buf_value.squeeze(1))
        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        return buf_r_sum, buf_advantage

    @staticmethod
    def compute_reward_gae(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
        buf_advantage = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_sum = 0  # reward sum of previous step
        pre_advantage = 0  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]

            buf_advantage[i] = buf_reward[i] + buf_mask[i] * (pre_advantage - buf_value[i])  # fix a bug here
            pre_advantage = buf_value[i] + buf_advantage[i] * self.lambda_gae_adv

        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        return buf_r_sum, buf_advantage

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


def explore_env(prediction_1, prediction_2):
    trajectory_list1 = list()
    trajectory_list2 = list()

    episode_reward = np.zeros(N_PREDATOR)

    # Reset env
    [observations_row, observations_column, _] = env.reset(prediction_1.role, prediction_2.role, 3, 3,
                                                           prediction_1.punishment, prediction_2.punishment)
    step = 0
    pred1_state = observations_row
    pred2_state = observations_column

    pred1_initial_stack = [pred1_state for _ in range(FRAME_STACK_SIZE)]
    pred1_frame_stack = deque(pred1_initial_stack, maxlen=FRAME_STACK_SIZE)
    pred1_state = np.concatenate(pred1_frame_stack, axis=2)

    pred2_initial_stack = [pred2_state for _ in range(FRAME_STACK_SIZE)]
    pred2_frame_stack = deque(pred2_initial_stack, maxlen=FRAME_STACK_SIZE)
    pred2_state = np.concatenate(pred2_frame_stack, axis=2)

    # sample one trajectory
    while True:
        # Get actions
        pred1_action, pred1_prob = prediction_1.select_action(pred1_state)
        pred2_action, pred2_prob = prediction_2.select_action(pred2_state)

        pred1_action = int(pred1_action)
        pred2_action = int(pred2_action)
        actions = [pred1_action, pred2_action]
        # Take actions, observe next states and rewards
        [next_observations_row, next_observations_column, next_observations], reward_vectors, done, _ = env.step(
            actions)
        next_pred1_state = next_observations_row
        next_pred2_state = next_observations_column

        pred1_reward, pred2_reward = reward_vectors

        rewards = [pred1_reward, pred2_reward]
        # Store in replay buffers
        other1 = (pred1_reward, 0.0 if done else GAMMA, pred1_action, *pred1_prob)
        trajectory_list1.append((pred1_state, other1))

        other2 = (pred2_reward, 0.0 if done else GAMMA, pred2_action, *pred2_prob)
        trajectory_list2.append((pred2_state, other2))

        pred1_frame_stack.append(next_pred1_state)
        next_pred1_state = np.concatenate(pred1_frame_stack, axis=2)

        pred2_frame_stack.append(next_pred2_state)
        next_pred2_state = np.concatenate(pred2_frame_stack, axis=2)

        # Assign next state to current state !!
        pred1_state = next_pred1_state
        pred2_state = next_pred2_state

        step += 1
        episode_reward += np.array(rewards)
        if done:
            break

    prediction_1.reward_tracker.append(episode_reward[0])
    prediction_2.reward_tracker.append(episode_reward[1])

    return trajectory_list1, trajectory_list2, episode_reward[0], episode_reward[1]
