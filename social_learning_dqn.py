import numpy as np
import random
import torch
from Agent_DQN import AgentDQN
from collections import deque
from datetime import datetime
from env import IntersectionEnv

SEED = 20
IMAGE = True

BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 500

GAMMA = 0.99  # discount factor
ALPHA = 1e-2  # learning rate

TRAINING_EPISODES = 8000

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 1 / (2500 * 0.85)

COPY_TO_TARGET_EVERY = 1000  # Steps
START_TRAINING_AFTER = 50  # Episodes
MEAN_REWARD_EVERY = 300  # Episodes

INIT1 = 1
INIT2 = 1
FRAME_STACK_SIZE = 3

PATH_ID = f'/***/***'
PATH_DIR = './'
NUM_WEIGHTS = 2

N_PREDATOR = 2
N_agents = 10
env = IntersectionEnv(n_predator=N_PREDATOR, image=IMAGE)

if __name__ == '__main__':
    start_time = datetime.now()
    random.seed(SEED)
    np.random.seed(SEED)
    env.seed(SEED)

    # Initialise agents list
    agent_list = list()
    for i in range(N_agents):
        if i < 5:
            # if i == 0:
            #     punishment = -1
            # else:
            #     punishment = -5
            punishment = -5
            agent_list.append(
                [AgentDQN(0, 'row', 'cooperative', punishment), AgentDQN(1, 'column', 'cooperative', punishment)])
        else:
            # if i == 5:
            #     punishment = -5
            # else:
            #     punishment = -1
            punishment = -1
            agent_list.append(
                [AgentDQN(0, 'row', 'defective', punishment), AgentDQN(1, 'column', 'defective', punishment)])

    steps = 0

    for episode in range(1, TRAINING_EPISODES + 1):
        eps = max(EPSILON_START - episode * EPSILON_DECAY, EPSILON_END)

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

            # Reset env
            [observations_row, observations_column, observations] = env.reset(prediction_1.role, prediction_2.role,
                                                                              INIT1, INIT2, prediction_1.punishment,
                                                                              prediction_2.punishment)

            pred1_state = observations_row
            pred1_initial_stack = [pred1_state for _ in range(FRAME_STACK_SIZE)]
            pred1_frame_stack = deque(pred1_initial_stack, maxlen=FRAME_STACK_SIZE)
            pred1_state = np.concatenate(pred1_frame_stack, axis=2)  # State is now a stack of frames (5, 5, 9)

            pred2_state = observations_column
            pred2_initial_stack = [pred2_state for _ in range(FRAME_STACK_SIZE)]
            pred2_frame_stack = deque(pred2_initial_stack, maxlen=FRAME_STACK_SIZE)
            pred2_state = np.concatenate(pred2_frame_stack, axis=2)

            episode_reward = np.zeros(N_PREDATOR)

            # sample one trajectory
            while True:
                # Get actions
                pred1_action = prediction_1.select_action(pred1_state, eps)
                pred2_action = prediction_2.select_action(pred2_state, eps)

                actions = [pred1_action, pred2_action]

                # Take actions, observe next states and rewards
                [next_observations_row, next_observations_column,
                 next_observations], rewards, done, _ = env.step(
                    actions)

                next_pred1_state = next_observations_row
                next_pred2_state = next_observations_column
                pred1_reward, pred2_reward = rewards

                # Store in replay buffers
                pred1_frame_stack.append(next_pred1_state)
                next_pred1_state = np.concatenate(pred1_frame_stack, axis=2)
                prediction_1.replay_memory.append((pred1_state, pred1_action, pred1_reward, next_pred1_state, done))

                pred2_frame_stack.append(next_pred2_state)
                next_pred2_state = np.concatenate(pred2_frame_stack, axis=2)
                prediction_2.replay_memory.append((pred2_state, pred2_action, pred2_reward, next_pred2_state, done))

                pred1_state = next_pred1_state
                pred2_state = next_pred2_state

                steps += 1
                episode_reward += np.array(rewards)

                if done:
                    break

                if steps % COPY_TO_TARGET_EVERY == 0:
                    prediction_1.update_target_model()
                    prediction_2.update_target_model()

            prediction_1.reward_tracker.append(episode_reward[0])
            prediction_2.reward_tracker.append(episode_reward[1])

            ep_rewards = [np.round(episode_reward[0], 2),
                          np.round(episode_reward[1], 2)]
            av_rewards = [np.round(prediction_1.reward_tracker.mean(), 2),
                          np.round(prediction_2.reward_tracker.mean(), 2)]

            if episode % 100 == 0:
                print(
                    "\rEpisode: {}, Trail: {}, Time: {}, Reward1: {}, Reward2: {}, Avg Reward1 {}, "
                    "Avg Reward2 {}, eps: {:.3f}\n".format(episode, trail_s, datetime.now() - start_time,
                                                           ep_rewards[0], ep_rewards[1], av_rewards[0],
                                                           av_rewards[1], eps), end="")

            if episode > START_TRAINING_AFTER:  # Wait for buffer to fill up a bit
                prediction_1.training_step()
                prediction_2.training_step()

        if episode % 250 == 0:
            for i in range(N_agents):
                p = agent_list[i][0].punishment
                torch.save(agent_list[i][0].dqn.state_dict(), f'{PATH_DIR}/data/{PATH_ID}_{i}_row_{p}.pth')
                torch.save(agent_list[i][1].dqn.state_dict(), f'{PATH_DIR}/data/{PATH_ID}_{i}_column_{p}.pth')
                agent_list[i][0].plot_learning_curve(
                    image_path=f'{PATH_DIR}/data/{PATH_ID}_{i}_row_{p}.png',
                    csv_path=f'{PATH_DIR}/data/{PATH_ID}_{i}_row_{p}.csv')
                agent_list[i][1].plot_learning_curve(
                    image_path=f'{PATH_DIR}/data/{PATH_ID}_{i}_column_{p}.png',
                    csv_path=f'{PATH_DIR}/data/{PATH_ID}_{i}_column_{p}.csv')

    for i in range(N_agents):
        p = agent_list[i][0].punishment
        torch.save(agent_list[i][0].dqn.state_dict(), f'{PATH_DIR}/data/{PATH_ID}_{i}_row_{p}.pth')
        torch.save(agent_list[i][1].dqn.state_dict(), f'{PATH_DIR}/data/{PATH_ID}_{i}_column_{p}.pth')
        agent_list[i][0].plot_learning_curve(
            image_path=f'{PATH_DIR}/data/{PATH_ID}_{i}_row_{p}.png',
            csv_path=f'{PATH_DIR}/data/{PATH_ID}_{i}_row_{p}.csv')
        agent_list[i][1].plot_learning_curve(
            image_path=f'{PATH_DIR}/data/{PATH_ID}_{i}_column_{p}.png',
            csv_path=f'{PATH_DIR}/data/{PATH_ID}_{i}_column_{p}.csv')

    run_time = datetime.now() - start_time
    print(f'\nRun time: {run_time} s')
