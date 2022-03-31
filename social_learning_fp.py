import numpy as np

import random
from Agent_FP import AgentFP
from collections import deque  # Used for replay buffer and reward tracking
from datetime import datetime  # Used for timing script
from env import IntersectionEnv


SEED = 24
IMAGE = True
BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 5000
GAMMA = 0.99  # discount factor
ALPHA = 1e-2  # learning rate
ETA = 0.1
TRAINING_EPISODES = 50000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 1 / (2500 * 0.85)
COPY_TO_TARGET_EVERY = 1000  # Steps
START_TRAINING_AFTER = 50  # Episodes
MEAN_REWARD_EVERY = 300  # Episodes
FRAME_STACK_SIZE = 3
PATH_DIR = './'
PATH_ID = 'basic_fp_social_learning_2'
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
        if i < 10:
            punishment = -5
            agent_list.append(
                [AgentFP(0, 'row', 'cooperative', punishment), AgentFP(1, 'column', 'cooperative', punishment)])
        else:
            punishment = -1
            agent_list.append(
                [AgentFP(0, 'row', 'defective', punishment), AgentFP(1, 'column', 'defective', punishment)])

    steps = 0

    for episode in range(1, TRAINING_EPISODES):  # training episode = 5000
        eps = max(EPSILON_START - episode * EPSILON_DECAY, EPSILON_END)

        ag = [id for id in range(N_agents)]
        for trail_s in range(1, int(N_agents / 2) + 1):  # each trail has two players random selected
            is_best_response = False
            if random.random() < ETA:
                is_best_response = True

            agent_i = random.choice(ag)
            ag.remove(agent_i)
            agent_j = random.choice(ag)
            ag.remove(agent_j)

            a_player = random.choice([0, 1])  # 0:row, 1: column
            if a_player == 0:
                pred1 = agent_list[agent_i][a_player]
                pred2 = agent_list[agent_j][1 - a_player]
            else:
                pred1 = agent_list[agent_j][1 - a_player]
                pred2 = agent_list[agent_i][a_player]

            # Reset env
            observations = env.reset(pred1.role, pred2.role, 3, 3, pred1.punishment, pred2.punishment)
            state, pred1_state, pred2_state = observations

            pred1_initial_stack = [pred1_state for _ in range(FRAME_STACK_SIZE)]
            pred1_frame_stack = deque(pred1_initial_stack, maxlen=FRAME_STACK_SIZE)
            pred1_state = np.concatenate(pred1_frame_stack, axis=2)  # State is now a stack of frames (5, 5, 9)

            pred2_initial_stack = [pred2_state for _ in range(FRAME_STACK_SIZE)]
            pred2_frame_stack = deque(pred2_initial_stack, maxlen=FRAME_STACK_SIZE)
            pred2_state = np.concatenate(pred2_frame_stack, axis=2)  # State is now a stack of frames

            episode_reward = np.zeros(N_PREDATOR)

            # sample one trajectory
            while True:
                # Get actions
                pred1_action = pred1.select_action(pred1_state, eps, is_best_response)
                pred2_action = pred2.select_action(pred2_state, eps, is_best_response)

                actions = [pred1_action, pred2_action]

                # Take actions, observe next states and rewards
                next_observations, reward_vectors, done, _ = env.step(actions)
                state, next_pred1_state, next_pred2_state = next_observations

                pred1_reward, pred2_reward = reward_vectors

                rewards = [pred1_reward, pred2_reward]

                # Store in replay buffers
                pred1_frame_stack.append(next_pred1_state)
                next_pred1_state = np.concatenate(pred1_frame_stack, axis=2)
                pred1.replay_memory.append((pred1_state, pred1_action, pred1_reward, next_pred1_state, done))
                pred2_frame_stack.append(next_pred2_state)
                next_pred2_state = np.concatenate(pred2_frame_stack, axis=2)
                pred2.replay_memory.append((pred2_state, pred2_action, pred2_reward, next_pred2_state, done))

                if is_best_response:
                    pred1.reservoir_buffer.push(pred1_state, pred1_action)
                    pred2.reservoir_buffer.push(pred2_state, pred2_action)

                # Assign next state to current state !!
                pred1_state = next_pred1_state
                pred2_state = next_pred2_state

                steps += 1

                episode_reward += np.array(rewards)

                if done:
                    break

                if steps % COPY_TO_TARGET_EVERY == 0:  # COPY_TO_TARGET_EVERY = 1000
                    pred1.update_target_model()
                    pred2.update_target_model()

            # print(episode_reward)
            pred1.reward_tracker.append(episode_reward[0])
            pred2.reward_tracker.append(episode_reward[1])

            ep_rewards = [np.round(episode_reward[0], 2),
                          np.round(episode_reward[1], 2)]
            av_rewards = [np.round(pred1.reward_tracker.mean(), 2),
                          np.round(pred2.reward_tracker.mean(), 2)]

            if episode % 100 == 0:
                print(
                    "\rEpisode: {}, Trail: {}, Time: {}, Reward1: {}, Reward2: {}, Avg Reward1 {}, "
                    "Avg Reward2 {}, eps: {:.3f}\n".format(episode, trail_s, datetime.now() - start_time,
                                                           ep_rewards[0], ep_rewards[1], av_rewards[0],
                                                           av_rewards[1], eps), end="")

            if len(pred1.reservoir_buffer) > 64 and len(pred2.reservoir_buffer) > 64:
                pred1.training_step()
                pred2.training_step()

            if episode % 250 == 0:
                for i in range(N_agents):
                    # agent_list[i].model.save(f'{PATH_DIR}/final1/{PATH_ID}_{i}.h5')
                    agent_list[i][0].plot_learning_curve(
                        image_path=f'{PATH_DIR}/final/{PATH_ID}_{i}_row.png',
                        csv_path=f'{PATH_DIR}/final/{PATH_ID}_{i}_row.csv')
                    agent_list[i][1].plot_learning_curve(
                        image_path=f'{PATH_DIR}/final/{PATH_ID}_{i}_column.png',
                        csv_path=f'{PATH_DIR}/final/{PATH_ID}_{i}_column.csv')

    for i in range(N_agents):
        # agent_list[i].model.save(f'{PATH_DIR}/final1/{PATH_ID}_{i}.h5')
        agent_list[i][0].plot_learning_curve(
            image_path=f'{PATH_DIR}/final/{PATH_ID}_{i}_row.png',
            csv_path=f'{PATH_DIR}/final/{PATH_ID}_{i}_row.csv')
        agent_list[i][1].plot_learning_curve(
            image_path=f'{PATH_DIR}/final/{PATH_ID}_{i}_column.png',
            csv_path=f'{PATH_DIR}/final/{PATH_ID}_{i}_column.csv')

    run_time = datetime.now() - start_time
    print(f'\nRun time: {run_time} s')



