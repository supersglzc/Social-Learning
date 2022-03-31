import os
import numpy as np
import matplotlib.pyplot as plt
import random
import gym


class IntersectionBackgroundEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    CAPTURE_RADIUS = 3
    MAX_STEPS = 150
    intersection_location = [3, 3]
    norm_reward = 2

    def __init__(self, n_predator=2, image=True):
        super(IntersectionBackgroundEnv, self).__init__()

        self.n_predator = n_predator
        self.image = image
        self.config = Config()

        self.base_gridmap_array = self._load_map()
        self.base_gridmap_image = self._to_image(self.base_gridmap_array)

        self.observation_shape = self.base_gridmap_image.shape
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=self.observation_shape)

        self.action_space = gym.spaces.Discrete(len(self.config.action_dict_0))
        self.crash_1 = 0

    @staticmethod
    def _load_map():
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "map.txt")
        with open(path, 'r') as f:
            gridmap = f.readlines()

        gridmap_array = np.array(
            list(map(lambda x: list(map(lambda y: int(y), x.split(' '))), gridmap)))
        return gridmap_array

    def _to_image(self, gridmap_array):
        image = np.zeros((gridmap_array.shape[0], gridmap_array.shape[1], 3), dtype=np.float32)

        for row in range(gridmap_array.shape[0]):
            for col in range(gridmap_array.shape[1]):
                grid = gridmap_array[row, col]

                if grid == self.config.grid_dict["empty"]:
                    image[row, col] = self.config.color_dict["empty"]
                elif grid == self.config.grid_dict["wall"]:
                    image[row, col] = self.config.color_dict["wall"]
                elif grid == self.config.grid_dict["prey"]:
                    image[row, col] = self.config.color_dict["prey"]
                elif grid == self.config.grid_dict["predator"]:
                    image[row, col] = self.config.color_dict["predator"]
                else:
                    raise ValueError()

        return image

    def _reset_agents(self, role1, role2):
        self.crash_1 = 0
        self.agents = []

        locs_row = [np.array([3, 0]), np.array([3, 1]), np.array([3, 2])]
        random.shuffle(locs_row)
        locs_background = [np.array([1, 3]), np.array([2, 3])]
        random.shuffle(locs_background)
        locs_column = [np.array([0, 3])]
        if locs_background[0][0] != 1:
            locs_column.append(np.array([1, 3]))
        random.shuffle(locs_column)

        agent = Agent(0, "predator", self.base_gridmap_array, 5, locs_row[0], role1)
        self.agents.append(agent)
        agent = Agent(1, "predator", self.base_gridmap_array, 5, locs_column[0], role2)
        self.agents.append(agent)
        agent = Agent(2, "predator", self.base_gridmap_array, 5, locs_background[0], "background")
        self.agents.append(agent)

    def _render_gridmap(self):
        gridmap_image = np.copy(self.base_gridmap_image)

        for agent in self.agents:
            if agent.hit_wall:
                continue
            else:
                location = agent.location
                gridmap_image[location[0], location[1]] = self.config.color_dict[agent.type]

        return gridmap_image

    def step(self, actions):
        for i in range(len(self.agents)):
            if i < 2:
                action = actions[i]
            else:
                action = 1

                # action = 1

            if self.agents[i].initial_l[0] == 3:
                action = list(self.config.action_dict_0.keys())[action]
                next_location = self.agents[i].location + self.config.action_dict_0[action]
            else:
                action = list(self.config.action_dict_1.keys())[action]
                next_location = self.agents[i].location + self.config.action_dict_1[action]

            self.agents[i].location = next_location

        gridmap_image = self._render_gridmap()

        observations = list()
        for agent in self.agents[:2]:
            observation = self._get_observation(agent, gridmap_image.copy())
            observations.append(observation)
        observations.append(self._get_observation_full(gridmap_image.copy()))

        hunted_predator = 0
        for predator in self.agents:
            if predator.hit_wall:
                hunted_predator += 1

        crash = None
        crash_location = None
        all_locations = []
        for agent in self.agents:
            if agent.location.tolist() in all_locations:
                crash = 1
                self.crash_1 = 1
                crash_location = agent.location

            else:
                all_locations.append(agent.location.tolist())

        rewards = [-2, -2]

        for i in range(2):
            if self.agents[i].hit_wall:
                rewards[i] = 0

        if crash is not None:
            for i in range(len(self.agents)):
                if np.array_equal(self.agents[i].location, crash_location):
                    if self.agents[i].role == "cooperative":
                        rewards[i] = -10
                    elif self.agents[i].role == "defective":
                        rewards[i] = 10

        if hunted_predator == 3:
            rewards = [self.norm_reward, self.norm_reward]
        self.n_steps += 1

        if (hunted_predator == 3) or (self.n_steps >= self.MAX_STEPS) or (crash is not None):
            done = True
        else:
            done = False

        return observations, rewards, done, {}

    def reset(self, role1, role2):
        self.n_steps = 0
        self._reset_agents(role1, role2)
        gridmap_image = self._render_gridmap()
        observations = []

        for agent in self.agents[:2]:
            observation = self._get_observation(agent, gridmap_image.copy())
            observations.append(observation)
        observations.append(self._get_observation_full(gridmap_image.copy()))

        return observations

    def _get_observation_full(self, gridmap_image):
        for agent in self.agents:
            location = agent.location
            if agent.role == "defective" and agent.hit_wall is False:
                gridmap_image[location[0], location[1]] = self.config.color_dict["defective"]
            elif agent.role == "background" and agent.hit_wall is False:
                gridmap_image[location[0], location[1]] = self.config.color_dict["background"]
        observation = gridmap_image.copy()
        return observation

    def _get_observation(self, agent, gridmap_image):
        observation = gridmap_image.copy()
        if agent.type == 'prey':
            return observation
        else:
            for agent_x in self.agents:
                location = agent_x.location
                if agent_x.role == "defective" and agent_x.hit_wall is False:
                    observation[location[0], location[1]] = self.config.color_dict['defective']
                elif agent_x.role == "background" and agent_x.hit_wall is False:
                    observation[location[0], location[1]] = self.config.color_dict['background']

                if agent_x.role == agent.role and agent_x.id != agent.id and agent_x.hit_wall is False:
                    observation[location[0], location[1]] = self.config.color_dict['other_agent']

            return observation

    def render(self, mode='human', close=False):
        gridmap_image = self._render_gridmap()

        plt.figure(1)
        plt.clf()
        plt.imshow(gridmap_image)
        plt.axis('off')
        plt.pause(0.00001)


class Config(object):
    def __init__(self):
        super(Config, self).__init__()

        self._set_action_dict()
        self._set_grid_dict()
        self._set_color_dict()

    def _set_action_dict(self):
        self.action_dict_0 = {
            "stay": np.array([0, 0]),
            "move_forward": np.array([0, 1]),
        }
        self.action_dict_1 = {
            "stay": np.array([0, 0]),
            "move_forward": np.array([1, 0]),
        }

    def _set_grid_dict(self):
        self.grid_dict = {
            "empty": 0,
            "wall": 1,
            "prey": 2,
            "predator": 3,
        }

    def _set_color_dict(self):
        self.color_dict = {
            "empty": [0., 0., 0.],  # Black
            "wall": [0.5, 0.5, 0.5],  # Gray
            "background": [1., 0., 0.],  # Red
            "predator": [0., 0., 1.],  # Blue
            "defective": [0., 1., 0.],  # Green
            "other_agent": [0., 0., 0.5],  # Light Blue
        }


class Agent(object):
    def __init__(self, i_agent, agent_type, base_gridmap_array, agent_radius, ini_loc, role):
        self.id = i_agent
        self.type = agent_type
        self.base_gridmap_array = base_gridmap_array
        self.radius = agent_radius
        self.initial_l = ini_loc
        self.hit_wall = False
        self.role = role

        self.config = Config()
        self._location = self._reset_location()

    def _reset_location(self):
        location = self.initial_l  # origin
        return location

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, value):
        if (0 <= value[0] < self.base_gridmap_array.shape[0]) and (0 <= value[1] < self.base_gridmap_array.shape[1]):
            grid = self.base_gridmap_array[value[0], value[1]]
            if grid != self.config.grid_dict["wall"]:
                self._location = value
            else:
                self.hit_wall = True
                self._location = value
        else:
            self.hit_wall = True
            self._location = value


if __name__ == '__main__':
    N_PREDATOR = 4
    env = IntersectionBackgroundEnv(n_predator=N_PREDATOR, image=True)
    observations = env.reset("cooperative", "cooperative")  # env.render()
    plt.figure(2)
    plt.cla()
    plt.imshow(observations[1])
    plt.axis('off')
    plt.pause(2)
    a = [[0, 1], [0, 1], [0, 1], [0, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]

    for i in range(env.MAX_STEPS):
        prey_actions = [np.random.choice([0, env.action_space.sample()], p=[0.8, 0.2])]
        pred_actions = [env.action_space.sample() for _ in range(N_PREDATOR)]
        actions = pred_actions
        observations, rewards, done, _ = env.step(a[i])  # env.render()
        print(rewards)
        plt.figure(2)
        plt.cla()
        plt.imshow(observations[1])
        plt.axis('off')
        plt.pause(2)

        if done:
            break
