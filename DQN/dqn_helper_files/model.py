import torch.nn as nn


class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    Nature DQN paper.
    """

    def __init__(self, observation_space, action_space):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """

        # TODO Implement DQN Network
        self.conv1 = nn.Conv2d(in_channels=observation_space.n, out_channels=32, kernel_size=8, stride=4)	
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=action_space.n)
        #raise NotImplementedError

    def forward(self, x):
        # TODO Implement forward pass
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x.view(x.size(0), -1))
