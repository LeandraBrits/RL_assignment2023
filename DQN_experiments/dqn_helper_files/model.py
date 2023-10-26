from gym import spaces
import torch.nn as nn
import torch
import torch.nn.functional as F

class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    Nature DQN paper.
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

        """
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        I 
        DONT
        TRUST
        WHAT
        I
        DID
        BELOW
        AT
        ALL

        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        # defining the layers of our neural network

        # conv stands for convolutional

        self.conv1 = nn.Conv2d(observation_space.shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # we're making "fake" data so that we can do a forward pass on it and use the
        # self._to_linear variable to get the shape of the input data
        # inspired by sentdex: https://pythonprogramming.net/convnet-model-deep-learning-neural-network-pytorch/

        x = torch.zeros(1, *observation_space.shape)
        self._to_linear = None
        self.convs(x)

        #fc stands for fully connected.

        self.fc1 = nn.Linear(self._to_linear,  out_features=512) # flattening
        self.fc2 = nn.Linear(in_features=512, out_features=action_space.n) # 512 in, number of actions out (because the actions are our classes)

        #raise NotImplementedError

    def convs(self, x):

        x = F.Relu(self.conv1(x))
        x = F.Relu(self.conv2(x))
        x = F.Relu(self.conv3(x))
        
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
    
    def forward(self, x):
        # implements forward pass through neural network
        x= self.convs(x)
        x = x.view(-1, self._to_linear)  # reshaping
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x) # bc this is our output layer,  no activation is required
        return x 

        raise NotImplementedError
