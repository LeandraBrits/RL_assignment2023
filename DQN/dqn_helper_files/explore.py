import numpy as np
from nle import nethack
import random
from minihack import RewardManager
import minihack
import torch
import matplotlib.pyplot as plt

from dqn_helper_files.replay_buffer import ReplayBuffer
from dqn_helper_files.wrappers import *
from dqn_helper_files.agent import DQNAgent

class ExploreEvent(minihack.reward_manager.Event):
    def __init__(self, reward: float, repeatable: bool, terminal_required: bool, terminal_sufficient: bool):
        super().__init__(reward, repeatable, terminal_required, terminal_sufficient)

    def check(self, env, previous_observation, action, observation) -> float:
        # blank spots are 32
        # agent is 64
        # agent spawn point is 60
        # pathways are 35
        # obs[1] is the char observation 
        # print("+++++++++++++++++++\nobs = \n++++++++++++++++++++++\n", observation[1])
        current = sum(np.count_nonzero(i == 35) for i in observation[1])
        current += sum(np.count_nonzero(i == 60) for i in observation[1])
        prev = sum(np.count_nonzero(i == 35) for i in previous_observation[1])
        prev += sum(np.count_nonzero(i == 60) for i in previous_observation[1])
        if current > prev:
            return self.reward
        else:
            return 0