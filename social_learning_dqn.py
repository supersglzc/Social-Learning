import numpy as np
import random
from Agent_DQN import AgentDQN
from collections import deque
from datetime import datetime
from env import IntersectionEnv
import pandas as pd

SEED = 20
IMAGE = True

BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 500

GAMMA = 0.99  # discount factor
ALPHA = 1e-2  # learning rate

TRAINING_EPISODES = 5000

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 1 / (2500 * 0.85)

COPY_TO_TARGET_EVERY = 1000  # Steps
START_TRAINING_AFTER = 50  # Episodes
MEAN_REWARD_EVERY = 300  # Episodes

FRAME_STACK_SIZE = 3

PATH_ID = f'dqn'
PATH_DIR = '/Users/chenyuxiang/Desktop/4995DeepLearning/Social-Learning-main'
NUM_WEIGHTS = 2

N_PREDATOR = 4
N_agents = 20
env = IntersectionEnv(n_predator=N_PREDATOR, image=IMAGE)
sum_rewards_list = []
if __name__ == '__main__':
    start_time = datetime.now()
    random.seed(SEED)
    np.random.seed(SEED)
    env.seed(SEED)

    # Initialise agents list
    agent_list = list()

    for i in range(N_agents):
            punishment = -10
            agent_list.append(
                 [AgentDQN(0, '0', 'cooperative', punishment), AgentDQN(1, '1', 'defective', punishment),
                  AgentDQN(2, '2', 'cooperative', punishment), AgentDQN(3, '3', 'cooperative', punishment)])

    steps = 0

    for episode in range(1, TRAINING_EPISODES + 1):
        eps = max(EPSILON_START - episode * EPSILON_DECAY, EPSILON_END)
        ag = [id for id in range(N_agents)]
        for trail_s in range(1, int(N_agents / N_PREDATOR) + 1):  # each trail has two players random selected
            #INIT = np.random.choice(2, 4)
            INIT1 = 3
            INIT2 = 3
            INIT3 = 3
            INIT4 = 3
            agent_i = random.choice(ag)
            ag.remove(agent_i)
            agent_j = random.choice(ag)
            ag.remove(agent_j)
            agent_k = random.choice(ag)
            ag.remove(agent_k)
            agent_l = random.choice(ag)
            ag.remove(agent_l)
            prediction_1 = agent_list[agent_i][0]
            prediction_2 = agent_list[agent_j][1]
            prediction_3 = agent_list[agent_k][0]
            prediction_4 = agent_list[agent_l][1]

            # Reset env

            [observations1, observations2, observations3, observations4, observations] = env.reset(prediction_1.role,
                                                                                                   prediction_2.role,
                                                                                                   prediction_3.role,
                                                                                                   prediction_4.role,
                                                                              INIT1, INIT2, INIT3, INIT4,
                                                                              prediction_1.punishment,
                                                                              prediction_2.punishment,
                                                                              prediction_3.punishment,
                                                                              prediction_4.punishment)

            pred1_state = observations1
            pred1_initial_stack = [pred1_state for _ in range(FRAME_STACK_SIZE)]
            pred1_frame_stack = deque(pred1_initial_stack, maxlen=FRAME_STACK_SIZE)
            pred1_state = np.concatenate(pred1_frame_stack, axis=2)  # State is now a stack of frames (5, 5, 9)

            pred2_state = observations2
            pred2_initial_stack = [pred2_state for _ in range(FRAME_STACK_SIZE)]
            pred2_frame_stack = deque(pred2_initial_stack, maxlen=FRAME_STACK_SIZE)
            pred2_state = np.concatenate(pred2_frame_stack, axis=2)

            pred3_state = observations3
            pred3_initial_stack = [pred3_state for _ in range(FRAME_STACK_SIZE)]
            pred3_frame_stack = deque(pred3_initial_stack, maxlen=FRAME_STACK_SIZE)
            pred3_state = np.concatenate(pred3_frame_stack, axis=2)

            pred4_state = observations4
            pred4_initial_stack = [pred4_state for _ in range(FRAME_STACK_SIZE)]
            pred4_frame_stack = deque(pred4_initial_stack, maxlen=FRAME_STACK_SIZE)
            pred4_state = np.concatenate(pred4_frame_stack, axis=2)  # State is now a stack of frames (5, 5, 9)

            episode_reward = np.zeros(N_PREDATOR)

            # sample one trajectory
            while True:
                # Get actions
                pred1_action = prediction_1.select_action(pred1_state, eps)
                pred2_action = prediction_2.select_action(pred2_state, eps)
                pred3_action = prediction_3.select_action(pred3_state, eps)
                pred4_action = prediction_4.select_action(pred4_state, eps)


                actions = [pred1_action, pred2_action, pred3_action,pred4_action]

                # Take actions, observe next states and rewards
                [next_observations1, next_observations2,next_observations3,next_observations4,
                 next_observations], rewards, done, _ = env.step(
                    actions)

                next_pred1_state = next_observations1
                next_pred2_state = next_observations2
                next_pred3_state = next_observations3
                next_pred4_state = next_observations4

                pred1_reward, pred2_reward, pred3_reward, pred4_reward= rewards

                # Store in replay buffers
                pred1_frame_stack.append(next_pred1_state)
                next_pred1_state = np.concatenate(pred1_frame_stack, axis=2)
                prediction_1.replay_memory.append((pred1_state, pred1_action, pred1_reward, next_pred1_state, done))

                pred2_frame_stack.append(next_pred2_state)
                next_pred2_state = np.concatenate(pred2_frame_stack, axis=2)
                prediction_2.replay_memory.append((pred2_state, pred2_action, pred2_reward, next_pred2_state, done))

                pred3_frame_stack.append(next_pred3_state)
                next_pred3_state = np.concatenate(pred3_frame_stack, axis=2)
                prediction_3.replay_memory.append((pred3_state, pred3_action, pred3_reward, next_pred3_state, done))

                pred4_frame_stack.append(next_pred4_state)
                next_pred4_state = np.concatenate(pred4_frame_stack, axis=2)
                prediction_4.replay_memory.append((pred4_state, pred4_action, pred4_reward, next_pred4_state, done))

                pred1_state = next_pred1_state
                pred2_state = next_pred2_state
                pred3_state = next_pred3_state
                pred4_state = next_pred4_state


                steps += 1
                episode_reward += np.array(rewards)

                if done:
                    break

                if steps % COPY_TO_TARGET_EVERY == 0:
                    prediction_1.update_target_model()
                    prediction_2.update_target_model()
                    prediction_3.update_target_model()
                    prediction_4.update_target_model()


            prediction_1.reward_tracker.append(episode_reward[0])
            prediction_2.reward_tracker.append(episode_reward[1])
            prediction_3.reward_tracker.append(episode_reward[2])
            prediction_4.reward_tracker.append(episode_reward[3])

            ep_rewards = [np.round(episode_reward[0], 2),
                          np.round(episode_reward[1], 2),
                          np.round(episode_reward[2], 2),
                          np.round(episode_reward[3], 2), ]
            av_rewards = [np.round(prediction_1.reward_tracker.mean(), 2),
                          np.round(prediction_2.reward_tracker.mean(), 2),
                          np.round(prediction_3.reward_tracker.mean(), 2),
                          np.round(prediction_4.reward_tracker.mean(), 2), ]
            sum_rewards = sum(ep_rewards) / N_PREDATOR
            sum_rewards_list.append(sum_rewards)
            if episode % 100 == 0:
                print(
                    "\rEpisode: {}, Trail: {}, Time: {}, Sum_rewards: {}, eps: {:.3f}\n".format(episode, trail_s,
                                                                                           datetime.now() - start_time,

                                                                                           sum_rewards, eps), end="")
                # print(
                #     "\rEpisode: {}, Trail: {}, Time: {}, Reward1: {}, Reward2: {}, Reward3: {},  Avg Reward1 {},"
                #     "Avg Reward2 {}, Avg Reward3 {},  eps: {:.3f}\n".format(episode, trail_s, datetime.now() - start_time,
                #                                            ep_rewards[0], ep_rewards[1], ep_rewards[2], av_rewards[0],
                #                                            av_rewards[1],av_rewards[2], eps), end="")
                # print(
                #     "\rEpisode: {}, Trail: {}, Time: {}, Reward1: {}, Reward2: {}, Reward3: {}, Reward4: {}, Avg Reward1 {},"
                #     "Avg Reward2 {}, Avg Reward3 {}, Avg Reward4 {}, eps: {:.3f}\n".format(episode, trail_s, datetime.now() - start_time,
                #                                            ep_rewards[0], ep_rewards[1], ep_rewards[2], ep_rewards[3], av_rewards[0],
                #                                            av_rewards[1],av_rewards[2],av_rewards[3], eps), end="")

            if episode > START_TRAINING_AFTER:  # Wait for buffer to fill up a bit
                prediction_1.training_step()
                prediction_2.training_step()
                prediction_3.training_step()
                prediction_4.training_step()
    #     if episode % 250 == 0:
    #         for i in range(N_agents):
    #             p = agent_list[i][0].punishment
    #             torch.save(agent_list[i][0].dqn.state_dict(), f'{PATH_DIR}/data/{PATH_ID}_{i}_row_{p}.pth')
    #             torch.save(agent_list[i][1].dqn.state_dict(), f'{PATH_DIR}/data/{PATH_ID}_{i}_column_{p}.pth')
    #             agent_list[i][0].plot_learning_curve(
    #                 image_path=f'{PATH_DIR}/data/{PATH_ID}_{i}_row_{p}.png',
    #                 csv_path=f'{PATH_DIR}/data/{PATH_ID}_{i}_row_{p}.csv')
    #             agent_list[i][1].plot_learning_curve(
    #                 image_path=f'{PATH_DIR}/data/{PATH_ID}_{i}_column_{p}.png',
    #                 csv_path=f'{PATH_DIR}/data/{PATH_ID}_{i}_column_{p}.csv')
    #
    # for i in range(N_agents):
    #     p = agent_list[i][0].punishment
    #     torch.save(agent_list[i][0].dqn.state_dict(), f'{PATH_DIR}/data/{PATH_ID}_{i}_row_{p}.pth')
    #     torch.save(agent_list[i][1].dqn.state_dict(), f'{PATH_DIR}/data/{PATH_ID}_{i}_column_{p}.pth')
    #     agent_list[i][0].plot_learning_curve(
    #         image_path=f'{PATH_DIR}/data/{PATH_ID}_{i}_row_{p}.png',
    #         csv_path=f'{PATH_DIR}/data/{PATH_ID}_{i}_row_{p}.csv')
    #     agent_list[i][1].plot_learning_curve(
    #         image_path=f'{PATH_DIR}/data/{PATH_ID}_{i}_column_{p}.png',
    #         csv_path=f'{PATH_DIR}/data/{PATH_ID}_{i}_column_{p}.csv')
    run_time = datetime.now() - start_time
    df = pd.DataFrame(sum_rewards_list)
    df.to_csv('/Users/chenyuxiang/Desktop/4995DeepLearning/Social-Learning-main/DQN_{0}cars_{1}agents'.format(N_PREDATOR,N_agents))
    print(f'\nRun time: {run_time} s')
