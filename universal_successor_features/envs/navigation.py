import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import exputils as eu
import numpy as np
import mpi_sim


class MPINavigation(gym.Env):
    
    @staticmethod
    def default_config():
        """Define the default configuration of the simulation.
        This default configuration will be mixed with the configuration
        provided later.
        
        The default configuration is a 6x6 empty room with ARI in the middle. 
        The viewing area is 10x10 (note that the viewing area is just for graphical purposes
        and doesnt actually limit where ARI can go).

        Furthermore, the GUI is activated.
        """
        return eu.AttrDict(
            n_max_steps=5000,
            start_position = [-4., -4.],
            start_orientation = 0,
            goal_position = [4., 4.],
            goal_orientation = 0,
            goal_position_radius = 0.1,
            show_gui = False,
            max_real_time_factor = 0.0,
        )
        

    def __init__(self, config = None, **kwargs):
        
        #combine default configuration settings with custom ones and other possible args
        self.config = eu.combine_dicts(kwargs, config, self.default_config())

        self.cur_step = None

        self.processes_config = []
        if self.config.show_gui:
            self.processes_config.append({"type": "GUI"})

        #initialize the area of the map that the simulator will use
        self.visible_area = [[-10. , 10.],[-10. , 10]]
        self.simulation = mpi_sim.Simulation(
            visible_area = self.visible_area,
            max_real_time_factor = self.config.max_real_time_factor,
            objects = [
                {"type": "Wall", "id": 1, "position": [ 5.1, 0.0], "orientation": 0., "length": 10.},
                {"type": "Wall", "id": 2, "position": [-5.1, 0.0], "orientation": 0., "length": 10.},
                {"type": "Wall", "id": 3, "position": [ 0.0, 5.1], "orientation": np.pi/2, "length": 10.},
                {"type": "Wall", "id": 4, "position": [ 0.0,-5.1], "orientation": np.pi/2, "length": 10.},
            ],
            processes = self.processes_config
        )

        #create ARI
        self.agent = mpi_sim.objects.ARIRobot(
            components=[mpi_sim.objects.ARIRobot.default_mapping_config()],
            position=self.config.start_position,
            orientation=self.config.start_orientation,
        ) 
        self.simulation.add_object(self.agent)

        #insert furnitures here
        self.tables = []
        self.chairs = []
        
        #insert humans here
        self.humans = []
        self.human_goal = [] 


        #set resent checkpoint so that everytime reset is called for the simulator, it will reset to this point.
        self.simulation.set_reset_point()


        #define observation space
        self.observation_space = gym.spaces.Dict(
            occupancy_grid = gym.spaces.Box(
                low=0, high=1, shape=self.agent.mapping.local_map_shape
            ),
        )

        #define actin space
        self.action_space = gym.spaces.Discrete(4)

        #define reward range
        self.reward_range = (-np.inf, 0.)

    def reset(self, seed = None):

        #erase old humans
        self.humans = []
        self.human_goals = []

        self.cur_step = 0

        #set a reset point
        if not self.simulation.has_reset_point:
            self.simulation.set_reset_point()
        else:
            self.simulation.reset()

        #create humans with their start condition and goal position
        #
        #
        #
        ###########################################################


        #self.agent.box2d_body.position = self.config.start_position # Why do I need to do this?

        return None, None

    def step(self, action):
    
        reward = 0.0

        self.cur_step += 1

        # set ari's action:
            #   0: stop
            #   1: go straight
            #   2: go right
            #   3: go left 
        if action == 0:
            linear_velocity = 0.0
            angular_velocity = 0.0
        elif action == 1:
            linear_velocity = 1.0
            angular_velocity = 0.0
        elif action == 2:
            linear_velocity = 0.8
            angular_velocity = -1.0
        elif action == 3:
            linear_velocity = 0.8
            angular_velocity = 1.0
        else:
            raise ValueError(f"Unknown Action {action}")

        self.agent.forward_velocity = linear_velocity
        self.agent.orientation_velocity = angular_velocity

        #run a step of the simulation
        self.simulation.step()

        #create a list of the entities in the simulation with their position and orientation
        #the list contains agent and humans
        #"a" stands for agent and "h" stands for human. 
        entities_list = []
        entities_list.append(["a", np.array(self.agent.position), self.agent.orientation]) 

        #insert step part for humans
        #
        #
        #
        ############################

        observation = eu.AttrDict(
            occupancy_grid = self.agent.mapping.local_map,

        )

        # Set Reward
        distance_to_goal = mpi_sim.utils.measure_center_distance(self.agent, self.config.goal_position)
        reward += -1.0 * distance_to_goal


        #Conditions to terminate episode
        terminated = False
        if distance_to_goal <= self.config.goal_position_radius:
            terminated = True

        truncated = False
        if self.cur_step >= self.config.n_max_steps:
            truncated = True

        info = {}

        return observation, reward, terminated, truncated, info




if __name__ == '__main__':
    env = MPINavigation(show_gui = True, max_real_time_factor = 1)

    # episodes
    num_episodes = 3
    for _ in range(num_episodes):
        env.reset()

        # steps
        for _ in range(100):
            transition = env.step(env.action_space.sample())

