import os
import numpy as np
import matplotlib.pyplot as plt
import gym


class IntersectionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    CAPTURE_RADIUS = 3
    MAX_STEPS = 15
    intersection_location = [3, 3]
    norm_reward = 5

    def __init__(self, n_predator=4, image=True):
        super(IntersectionEnv, self).__init__()

        self.n_predator = n_predator
        self.image = image
        self.config = Config()
        self.punishment1 = None
        self.punishment2 = None
        self.punishment3 = None
        self.punishment4 = None
        self.crash = None
        self.c1 = None
        self.c2 = None
        self.c3 = None
        self.c4 = None
        self.base_gridmap_array = self._load_map()
        self.base_gridmap_image = self._to_image(self.base_gridmap_array)

        self.observation_shape = self.base_gridmap_image.shape
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=self.observation_shape)

        self.action_space = gym.spaces.Discrete(len(self.config.action_dict_0))

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

    def _reset_agents(self, role1, role2,role3,role4,init1, init2,init3,init4):
        self.crash_1 = 0
        self.crash = [0]*4
        self.agents = []
        self.c1 = None
        self.c2 = None
        self.c3 = None
        self.c4 = None

        locations = [np.array([3, 3 - init1]), np.array([3 - init2, 3]), np.array([4, 3 - init3]),np.array([3 - init3,4])]
        agent = Agent(0, "predator", self.base_gridmap_array, 5, locations[0], role1)
        self.agents.append(agent)
        agent = Agent(1, "predator", self.base_gridmap_array, 5, locations[1], role2)
        self.agents.append(agent)
        agent = Agent(2, "predator", self.base_gridmap_array, 5, locations[2], role3)
        self.agents.append(agent)
        agent = Agent(3, "predator", self.base_gridmap_array, 5, locations[3], role4)
        self.agents.append(agent)


    def _render_gridmap(self):
        gridmap_image = np.copy(self.base_gridmap_image)

        for agent in self.agents:
            location = agent.location
            gridmap_image[location[0], location[1]] = self.config.color_dict[agent.type]

        return gridmap_image

    def step(self, actions):
        for i in range(len(self.agents)):
            action = actions[i]
            if self.agents[i].initial_l[1] < 3:
                if self.crash[i] == 1:
                    action = list(self.config.action_dict_0.keys())[0]
                else:
                    action = list(self.config.action_dict_0.keys())[action]
                next_location = self.agents[i].location + self.config.action_dict_0[action]
            else:
                if self.crash[i] == 1:
                    action = list(self.config.action_dict_0.keys())[0]
                else:
                    action = list(self.config.action_dict_1.keys())[action]
                next_location = self.agents[i].location + self.config.action_dict_1[action]

            self.agents[i].location = next_location

        gridmap_image = self._render_gridmap()

        observations = list()
        for agent in self.agents[:4]:
            observation = self._get_observation(agent, gridmap_image.copy())
            observations.append(observation)
        observations.append(self._get_observation_full(gridmap_image.copy()))

        hunted_predator = None
        if self.agents[0].initial_l[0] == 3 and self.agents[1].initial_l[1] == 3 and self.agents[2].initial_l[0] == 3 and self.agents[3].initial_l[0] == 3:
            if (self.agents[0].location[1] == 5) and (self.agents[1].location[0] == 5) and (self.agents[2].location[1] == 5) and (self.agents[3].location[0] == 5):  # 10 is edge
                hunted_predator = 1
        #else:
            #if (self.agents[0].location[0] == 4) and (self.agents[1].location[1] == 4):  # 10 is edge
                #hunted_predator = 1
        all_locations = []
        n = len(self.agents)
        d = {}
        for i in range(n):
            d [i] = self.agents[i].location.tolist()
            m = len(all_locations)
            for j in range(m):
                if d[i]==all_locations[j]:
                    self.crash[i]=1
                    self.crash[j]=1
            else:
                all_locations.append(self.agents[i].location.tolist())

        rewards = [-1]*4

        for i in range(4):
            if self.agents[i].initial_l[1] < 3:
                if self.agents[i].location[1] == 5:
                    rewards[i] = 0
            else:
                if self.agents[i].location[0] == 5:
                    rewards[i] = 0
        if self.crash != [0]*3:
            if self.crash[0]==1:
                if self.c1 is None:
                    rewards[0] = self.punishment1
                    self.c1 = 1
                else:
                    rewards[0] = 0
            if self.crash[1]==1:
                if self.c2 is None:
                    rewards[1] = self.punishment2
                    self.c2 = 1
                else:
                    rewards[1] = 0
            if self.crash[2]==1:
                if self.c3 is None:
                    rewards[2] = self.punishment3
                    self.c3 = 1
                else:
                    rewards[2] = 0
            if self.crash[3]==1:
                if self.c4 is None:
                    rewards[3] = self.punishment4
                    self.c4 = 1
                else:
                    rewards[3] = 0

        if hunted_predator is not None:
            rewards = [self.norm_reward, self.norm_reward]
        self.n_steps += 1

        if (hunted_predator is not None) or (self.n_steps >= self.MAX_STEPS) or (self.crash ==[1, 1, 1,1]):
            done = True
        else:
            done = False
        # print(rewards,self.crash)
        return observations, rewards, done, {}

    def reset(self, role1, role2, role3,role4, init1, init2,init3, init4,punishment1, punishment2,punishment3,punishment4):
        self.n_steps = 0
        self.punishment1 = punishment1
        self.punishment2 = punishment2
        self.punishment3 = punishment3
        self.punishment4 = punishment4
        self._reset_agents(role1, role2,role3,role4,init1, init2,init3,init4)
        gridmap_image = self._render_gridmap()
        observations = []
        for agent in self.agents[:4]:
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
            #'move_left' : np.array([1, 0]),
            #'move_right': np.array([-1, 0]),

        }
        self.action_dict_1 = {
            "stay": np.array([0, 0]),
            "move_forward": np.array([1, 0]),
            #'move_left': np.array([0, -1]),
            #'move_right': np.array([0, 1]),
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
        else:
            self.hit_wall = True

# Test
if __name__ == '__main__':
    N_PREDATOR = 4
    env = IntersectionEnv(n_predator=N_PREDATOR, image=True)
    observations = env.reset("cooperative", "cooperative", "cooperative","cooperative", 3, 3, 3, 3,-10,-10, -10, -10)  # env.render()
    plt.figure(3)
    plt.cla()
    plt.imshow(observations[3])
    plt.axis('off')
    plt.pause(3)
    for i in range(env.MAX_STEPS):
        pred_actions = [env.action_space.sample() for _ in range(N_PREDATOR)]
        actions = pred_actions
        observations, rewards, done, _ = env.step(actions)  # env.render()
        plt.figure(3)
        plt.cla()
        plt.imshow(observations[3].squeeze())
        plt.axis('off')
        plt.pause(3)
        if done:
            break
