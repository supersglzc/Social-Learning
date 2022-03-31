import numpy as np
import random
import torch
from Agent_PPO import AgentPPO, explore_env
from datetime import datetime  # Used for timing script
from env import IntersectionEnv

SEED = 42
IMAGE = True
BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 6000
GAMMA = 0.99  # discount factor
ALPHA = 1e-2  # learning rate
TRAINING_EPISODES = 5000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 1 / (2500 * 0.85)
NUM_EPISODE = 150
COPY_TO_TARGET_EVERY = 1000  # Steps
START_TRAINING_AFTER = 50  # Episodes
MEAN_REWARD_EVERY = 300  # Episodes
FRAME_STACK_SIZE = 3
PATH_DIR = './'
PATH_ID = f'/***/***'
NUM_WEIGHTS = 2

N_PREDATOR = 2
N_agents = 10
env = IntersectionEnv(n_predator=N_PREDATOR, image=IMAGE)

if __name__ == '__main__':
    start_time = datetime.now()
    random.seed()
    np.random.seed(SEED)
    env.seed(SEED)

    # Initialise agents list
    agent_list = list()
    for i in range(N_agents):
        if i < 10:
            punishment = -5
            agent_list.append(
                [AgentPPO(0, 'row', 'cooperative', punishment), AgentPPO(1, 'column', 'cooperative', punishment)])
        else:
            punishment = -1
            agent_list.append(
                [AgentPPO(0, 'row', 'defective', punishment), AgentPPO(1, 'column', 'defective', punishment)])

    turn = True

    for episode in range(1, TRAINING_EPISODES):
        turn = True

        # empty replay buffer for on-policy algorithm
        for i in range(N_agents):
            for j in range(N_PREDATOR):
                agent_list[i][j].replay_buffer.empty_buffer()

        for k in range(NUM_EPISODE):
            ag = [id for id in range(N_agents)]

            for trail_s in range(1, int(N_agents / 2) + 1):  # each trail has two players random selected
                agent_i = random.choice(ag)
                ag.remove(agent_i)
                agent_j = random.choice(ag)
                ag.remove(agent_j)
                a_player = random.choice([0, 1])  # 0:row, 1: column
                if a_player == 0:
                    prediction_1 = agent_list[agent_i][a_player]
                    prediction_2 = agent_list[agent_j][1 - a_player]
                else:
                    prediction_1 = agent_list[agent_j][1 - a_player]
                    prediction_2 = agent_list[agent_i][a_player]

                trajectory_list1, trajectory_list2, r1, r2 = explore_env(prediction_1, prediction_2)

                prediction_1.replay_buffer.extend_buffer_from_list(trajectory_list1)
                prediction_2.replay_buffer.extend_buffer_from_list(trajectory_list2)

                ep_rewards = [np.round(r1, 2),
                              np.round(r2, 2)]
                av_rewards = [np.round(prediction_1.reward_tracker.mean(), 2),
                              np.round(prediction_2.reward_tracker.mean(), 2)]

                if turn:
                    print(
                        "\rEpisode: {}, Trail: {}, Row: {}, Column: {}, Time: {}, Reward1: {}, Reward2: {}, Avg Reward1 {}, "
                        "Avg Reward2 {}\n".format(episode, trail_s, agent_i, agent_j,
                                                  datetime.now() - start_time, ep_rewards[0],
                                                  ep_rewards[1], av_rewards[0], av_rewards[1]), end="")
            turn = False

        for i in range(N_agents):
            for j in range(N_PREDATOR):
                result = agent_list[i][j].update_net()

        for i in range(N_agents):
            torch.save(agent_list[i][0].act.state_dict(), f'{PATH_DIR}/{PATH_ID}_{i}_row.pth')
            torch.save(agent_list[i][1].act.state_dict(), f'{PATH_DIR}/{PATH_ID}_{i}_column.pth')
            agent_list[i][0].plot_learning_curve(
                image_path=f'{PATH_DIR}/{PATH_ID}_{i}_row.png',
                csv_path=f'{PATH_DIR}/{PATH_ID}_{i}_row.csv')
            agent_list[i][1].plot_learning_curve(
                image_path=f'{PATH_DIR}/{PATH_ID}_{i}_column.png',
                csv_path=f'{PATH_DIR}/{PATH_ID}_{i}_column.csv')
        print("Finish episode " + str(episode))

    run_time = datetime.now() - start_time
    print(f'\nRun time: {run_time} s')
