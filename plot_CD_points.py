import numpy as np
import random
import torch
from collections import deque  # Used for replay buffer and reward tracking
from Agent_DQN import AgentDQN, FRAME_STACK_SIZE
from env import IntersectionEnv
from social_learning_dqn import N_agents

env = IntersectionEnv(n_predator=2, image=True)


def play_game(ag1, ag2, role1, role2, init1, init2, punishment1, punishment2):
    eps = 0

    [observations_row, observations_column, observations] = env.reset(role1=role1, role2=role2, init1=init1,
                                                                      init2=init2, punishment1=punishment1,
                                                                      punishment2=punishment2)
    pred1_state = observations_row
    pred2_state = observations_column

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

    agent_list = dict()
    for i in range(1, 4):
        for j in range(1, 4):
            li = list()
            for k in range(N_agents):
                if k < N_agents / 2:
                    if k == 0:
                        punishment = -1
                    else:
                        punishment = -5
                    agent = [AgentDQN(0, 'row', 'cooperative', punishment),
                             AgentDQN(1, 'column', 'cooperative', punishment)]
                    agent[0].dqn.load_state_dict(torch.load(f"./data/CD_{i}{j}/2_{k}_row_{punishment}.pth"))
                    agent[1].dqn.load_state_dict(torch.load(f"./data/CD_{i}{j}/2_{k}_column_{punishment}.pth"))
                    li.append(agent)
                else:
                    if k == 5:
                        punishment = -5
                    else:
                        punishment = -1
                    agent = [AgentDQN(0, 'row', 'defective', punishment),
                             AgentDQN(1, 'column', 'defective', punishment)]
                    agent[0].dqn.load_state_dict(torch.load(f"./data/CD_{i}{j}/2_{k}_row_{punishment}.pth"))
                    agent[1].dqn.load_state_dict(torch.load(f"./data/CD_{i}{j}/2_{k}_column_{punishment}.pth"))
                    li.append(agent)
            agent_list[f'{i}{j}'] = li

    points_x = list()
    points_y = list()
    for i in range(5):
        print(i)
        for j in range(6):
            if j == 0:
                li = ['11']
            elif j == 1:
                li = ['22']
            elif j == 2:
                li = ['33']
            elif j == 3:
                li = ['13', '31']
            elif j == 4:
                li = ['23', '32']
            else:
                li = ['12', '21']

            C_C = [0, 0]
            c1 = 0
            C_D = [0, 0]
            c2 = 0
            D_C = [0, 0]
            c3 = 0
            D_D = [0, 0]
            c4 = 0

            for ag in li:
                for k in range(1000):
                    agent_i = random.choice(agent_list[ag])
                    agent_j = random.choice(agent_list[ag])
                    a_player = random.choice([0, 1])

                    if a_player == 0:
                        pred1 = agent_i[0]
                        pred2 = agent_j[1]
                    else:
                        pred1 = agent_j[0]
                        pred2 = agent_i[1]

                    role1 = pred1.role
                    role2 = pred2.role

                    if role1 == "cooperative" and role2 == "cooperative":
                        adder = C_C
                        c1 += 1
                    elif role1 == "cooperative" and role2 == "defective":
                        adder = C_D
                        c2 += 1
                    elif role1 == "defective" and role2 == "cooperative":
                        adder = D_C
                        c3 += 1
                    else:
                        adder = D_D
                        c4 += 1

                    r, c = play_game(pred1, pred2, role1, role2, int(ag[0]), int(ag[1]), pred1.p, pred2.p)

                    adder[0] += r[0]
                    adder[1] += r[1]

            R = (C_C[0] + C_C[1]) / (2 * c1)
            S = (C_D[0] + D_C[1]) / (c2 + c3)
            T = (C_D[1] + D_C[0]) / (c2 + c3)
            P = (D_D[0] + D_D[1]) / (2 * c4)

            if R > P and R > S and (2 * R > (T + S)) and ((T > R) or (P > S)):
                points_x.append(P - S)
                points_y.append(T - R)

    x = np.array(points_x).reshape(-1, 1)
    y = np.array(points_y).reshape(-1, 1)
    np.savetxt('./data/CD_points_8.csv', np.concatenate((x, y), axis=1), delimiter=",")
