from gym import spaces
import torch.nn as nn
import torch
import torch.nn.functional as F
import gym
from nle import nethack
import minihack
import numpy as np

class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network.
    """

    def __init__(self, observation_space: spaces.Box, action_space: spaces.Discrete):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        
        super().__init__() #initialises the parent class

        assert (
            type(observation_space) == spaces.Box
        ), "observation_space must be of type Box"
        assert (
            len(observation_space.shape) == 3
        ), "observation space must have the form channels x width x height"
        assert (
            type(action_space) == spaces.Discrete
        ), "action_space must be of type Discrete"
        
        self.conv1 = nn.Conv2d(observation_space.shape[0], 32, 6, stride=3)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 2, stride=1)
        
        # get the dimensions of the input by doing a forward pass with fake data

        input_size = self.get_input_size(observation_space.shape)

        self.fc1 = nn.Linear(input_size, 1024) #flattening
        self.fc2 = nn.Linear(1024, action_space.n) # 1024 in, number of actions out, bc the number of actions is the number of classes.

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)


    def get_input_size(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

