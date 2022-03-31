import numpy as np
import random
import torch
import matplotlib.pyplot as plt

from collections import deque  # Used for replay buffer and reward tracking

from utils import FRAME_STACK_SIZE
from Agent_DQN import AgentDQN
from env import IntersectionEnv

env = IntersectionEnv(n_predator=2, image=True)


def play_game(ag1, ag2, role1, role2, punishment1, punishment2):
    eps = 0

    [observations_row, observations_column, observations] = env.reset(role1=role1, role2=role2, init1=3, init2=3, punishment1=punishment1, punishment2=punishment2)
    pred1_state = observations_row
    pred2_state = observations_column
    plt.figure(2)
    plt.cla()
    plt.imshow(observations)
    plt.axis('off')
    plt.pause(1)

    # Pred 1
    pred1_initial_stack = [pred1_state for _ in range(FRAME_STACK_SIZE)]
    pred1_frame_stack = deque(pred1_initial_stack, maxlen=FRAME_STACK_SIZE)
    pred1_state = np.concatenate(pred1_frame_stack, axis=2)  # State is now a stack of frames
    # Pred 2
    pred2_initial_stack = [pred2_state for _ in range(FRAME_STACK_SIZE)]
    pred2_frame_stack = deque(pred2_initial_stack, maxlen=FRAME_STACK_SIZE)
    pred2_state = np.concatenate(pred2_frame_stack, axis=2)  # State is now a stack of frames

    episode_reward = np.zeros(2)
    steps = 0
    while True:
        pred1_action = ag1.select_action(pred1_state, eps)
        pred2_action = ag2.select_action(pred2_state, eps)
        actions = [pred1_action, pred2_action]

        [next_observations_row, next_observations_column, next_observations], reward_vectors, done, _ = env.step(actions)
        plt.figure(2)
        plt.cla()
        plt.imshow(next_observations)
        plt.axis('off')
        plt.pause(1)
        next_pred1_state = next_observations_row
        next_pred2_state = next_observations_column
        pred1_reward, pred2_reward = reward_vectors
        rewards = [pred1_reward, pred2_reward]

        # Pred 1
        pred1_frame_stack.append(next_pred1_state)
        next_pred1_state = np.concatenate(pred1_frame_stack, axis=2)

        # Pred 2
        pred2_frame_stack.append(next_pred2_state)
        next_pred2_state = np.concatenate(pred2_frame_stack, axis=2)

        # Assign next state to current state !!
        pred1_state = next_pred1_state
        pred2_state = next_pred2_state

        steps += 1
        episode_reward += np.array(rewards)

        if done:
            break
    return episode_reward, env.crash_1


if __name__ == '__main__':
    random.seed()
    env.seed()

    punishment = -5
    agent1 = [AgentDQN(0, 'row', 'cooperative', punishment), AgentDQN(1, 'column', 'cooperative', [punishment])]
    agent1[0].dqn.load_state_dict(torch.load(f"./data/CD_33/3_1_row.pth"))
    agent1[1].dqn.load_state_dict(torch.load(f"./data/CD_33/3_1_column.pth"))

    punishment = -1
    agent2 = [AgentDQN(0, 'row', 'defective', punishment), AgentDQN(1, 'column', 'defective', punishment)]
    agent2[0].dqn.load_state_dict(torch.load(f"./data/CD_33/3_2_row.pth"))
    agent2[1].dqn.load_state_dict(torch.load(f"./data/CD_33/3_2_column.pth"))

    for i in range(1000):
        a_player = random.choice([0, 1])

        if a_player == 0:
            pred1 = agent1[0]
            pred2 = agent2[1]
        else:
            pred1 = agent2[0]
            pred2 = agent1[1]

        role1 = pred1.role
        role2 = pred2.role

        r, c = play_game(pred1, pred2, role1, role2, pred1.punishment, pred2.punishment)


