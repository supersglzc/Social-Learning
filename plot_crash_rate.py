import numpy as np
import random
import torch
from collections import deque  # Used for replay buffer and reward tracking
from Agent_DQN import AgentDQN, FRAME_STACK_SIZE
from env import IntersectionEnv

env = IntersectionEnv(n_predator=2, image=True)


def play_game(ag1, ag2, role1, role2, init1, init2, punishment1, punishment2):
    eps = 0

    # Reset env
    [observations_row, observations_column, observations] = env.reset(role1=role1, role2=role2, init1=init1,
                                                                      init2=init2, punishment1=punishment1,
                                                                       punishment2=punishment2)
    pred1_state = observations_row
    pred2_state = observations_column

    # Create deque for storing stack of N frames
    pred1_initial_stack = [pred1_state for _ in range(FRAME_STACK_SIZE)]
    pred1_frame_stack = deque(pred1_initial_stack, maxlen=FRAME_STACK_SIZE)
    pred1_state = np.concatenate(pred1_frame_stack, axis=2)  # State is now a stack of frames

    pred2_initial_stack = [pred2_state for _ in range(FRAME_STACK_SIZE)]
    pred2_frame_stack = deque(pred2_initial_stack, maxlen=FRAME_STACK_SIZE)
    pred2_state = np.concatenate(pred2_frame_stack, axis=2)  # State is now a stack of frames

    episode_reward = np.zeros(2)
    steps = 0
    while True:
        pred1_action = ag1.select_action(pred1_state, eps)
        pred2_action = ag2.select_action(pred2_state, eps)
        actions = [pred1_action, pred2_action]

        [next_observations_row, next_observations_column, next_observations], reward_vectors, done, _ = env.step(
            actions)

        next_pred1_state = next_observations_row
        next_pred2_state = next_observations_column

        pred1_reward, pred2_reward = reward_vectors
        rewards = [pred1_reward, pred2_reward]

        # Store in replay buffers
        pred1_frame_stack.append(next_pred1_state)
        next_pred1_state = np.concatenate(pred1_frame_stack, axis=2)

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

    agent_list = list()
    for i in range(11):
        buf = list()
        for j in range(11):
            li = list()
            for _ in range(i):
                k = random.choice([5, 6, 7, 8, 9])
                if k == 5:
                    punishment = -5
                else:
                    punishment = -1
                agent = [AgentDQN(0, 'row', 'defective', punishment), AgentDQN(1, 'column', 'defective', punishment)]
                agent[0].dqn.load_state_dict(torch.load(f"./data/CD_33/2_{k}_row_{punishment}.pth"))
                agent[1].dqn.load_state_dict(torch.load(f"./data/CD_33/2_{k}_column_{punishment}.pth"))
                li.append(agent)
            for _ in range(j):
                k = random.choice([0, 1, 2, 3, 4])
                if k == 0:
                    punishment = -1
                else:
                    punishment = -5
                agent = [AgentDQN(0, 'row', 'cooperative', punishment),
                         AgentDQN(1, 'column', 'cooperative', punishment)]
                agent[0].dqn.load_state_dict(torch.load(f"./data/CD_33/2_{k}_row_{punishment}.pth"))
                agent[1].dqn.load_state_dict(torch.load(f"./data/CD_33/2_{k}_column_{punishment}.pth"))
                li.append(agent)
            buf.append(li)
        agent_list.append(buf)

    crash_rate = list()
    for i in range(11):
        buf = list()
        for j in range(11):
            ag = agent_list[i][j]
            crash = 0
            for k in range(1000):
                if len(ag) == 0:
                    continue
                agent_i = random.choice(ag)
                agent_j = random.choice(ag)
                a_player = random.choice([0, 1])

                if a_player == 0:
                    pred1 = agent_i[0]
                    pred2 = agent_j[1]
                else:
                    pred1 = agent_j[0]
                    pred2 = agent_i[1]

                role1 = pred1.role
                role2 = pred2.role

                r, c = play_game(pred1, pred2, role1, role2, 3, 3, pred1.p, pred2.p)
                crash += c

            crash /= 3000
            buf.append(crash)
        crash_rate.append(buf)
    rate = np.asarray(crash_rate)
    np.savetxt('./data/crash_rate.csv', rate, delimiter=",")
