import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from utils import MovingAverage

colour_palette = get_cmap(name='Set1').colors
MEAN_REWARD_EVERY = 500
fig, ax = plt.subplots()
ax.clear()

k = 10000
y = list()
for i in range(10):
    reward_data = np.loadtxt(f'./data/basic_ppo_10/2_{i}_row.csv', delimiter=",")
    if i == 0:
        # x = reward_data[:, 0]
        y = reward_data[:, 1][:k]
    else:
        for j in range(k):
            y[j] += reward_data[:, 1][j]

for i in range(10):
    reward_data = np.loadtxt(f'./data/basic_ppo_10/2_{i}_column.csv', delimiter=",")
    for j in range(k):
        y[j] += reward_data[:, 1][j]

for j in range(len(y)):
    y[j] = y[j] / 20

episodes = np.array([i for i in range(len(y))]).reshape(-1, 1)
rewards = np.array(y).reshape(-1, 1)
np.savetxt('./data/basic_ppo_social_learning_10.csv', np.concatenate((episodes, rewards), axis=1), delimiter=",")

reward_data = np.loadtxt('./data/basic_ppo_social_learning_10.csv', delimiter=",")
x = reward_data[:, 0]
tracker = MovingAverage(maxlen=MEAN_REWARD_EVERY)
mean_rewards = np.zeros(len(reward_data))
for i, (_, reward) in enumerate(reward_data):
    tracker.append(reward)
    mean_rewards[i] = tracker.mean()

# Create plot
# ax.plot(x, y, alpha=0.2, c=colour_palette[0])
ax.plot(x[MEAN_REWARD_EVERY // 2:], mean_rewards[MEAN_REWARD_EVERY // 2:], c=colour_palette[0])
ax.set_xlabel('episode')
ax.set_ylabel('reward per episode')
ax.grid(True, ls=':')

fig.savefig('./data/basic_ppo_social_learning_10.png')
