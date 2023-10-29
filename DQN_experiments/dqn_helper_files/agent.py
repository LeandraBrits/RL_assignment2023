from gym import spaces
import numpy as np
import torch
import random
import gym
from nle import nethack
import minihack
import numpy as np

from dqn_helper_files.model import DQN
from dqn_helper_files.replay_buffer import ReplayBuffer

device = torch.device("cuda:0")

class DQNAgent:
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        replay_buffer: ReplayBuffer,
        use_double_dqn,
        lr,
        batch_size,
        gamma,
    ):
        """
        Initialise the DQN algorithm using the Adam optimiser
        
        :param action_space: the action space of the environment
        :param observation_space: the state space of the environment
        :param replay_buffer: storage for experience replay
        :param lr: the learning rate for Adam
        :param batch_size: the batch size
        :param gamma: the discount factor
        """
        # TODO: Initialise agent's networks, optimiser and replay buffer

        self.action_space = action_space
        self.observation_space = observation_space

        # agent's replay buffer
        self.replay_buffer=replay_buffer
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.use_double_dqn=use_double_dqn    

        #agent's networks
        
        self.Q_current=DQN(observation_space, action_space)
        self.Q_targets=DQN(observation_space, action_space)

        # agent's loss optimiser, which adjusts the neural network parameters to minimise loss 

        self.optimizer=torch.optim.Adam(self.Q_current.parameters(), lr=lr)

    

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        # TODO
        #   Optimise the TD-error over a single minibatch of transitions
        #   Sample the minibatch from the replay-memory
        #   using done (as a float) instead of if statement
        #   return loss


        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = np.array(states) 
        next_states = np.array(next_states) 
        states = torch.from_numpy(states).float().to(self.Q_current.device)
        actions = torch.from_numpy(actions).long().to(self.Q_current.device)
        rewards = torch.from_numpy(rewards).float().to(self.Q_current.device)
        next_states = torch.from_numpy(next_states).float().to(self.Q_current.device)
        dones = torch.from_numpy(dones).float().to(self.Q_current.device)

        with torch.no_grad():
            if self.use_double_dqn:
                _, max_next_action = self.Q_current(next_states).max(1)
                max_next_q_values = self.Q_targets(next_states).gather(1, max_next_action.unsqueeze(1)).squeeze()
            else:
                next_q_values = self.Q_targets(next_states)
                max_next_q_values, _ = next_q_values.max(1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        input_q_values = self.Q_current(states)
        input_q_values = input_q_values.gather(1, actions.unsqueeze(1)).squeeze()

        loss = torch.nn.functional.smooth_l1_loss(input_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        del states
        del next_states
        return loss.item()
        
        raise NotImplementedError

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        # TODO update target_network parameters with policy_network parameters

        self.Q_targets.load_state_dict(self.Q_current.state_dict())
        

    def save(self, file_name):
        torch.save(self.policy_network, file_name)

    def act(self, state: np.ndarray):
        """
        
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        
        """
        
        # TODO Select action greedily from the Q-network given the state

        # return action
        state = np.array(state)
        state=torch.from_numpy(state).float().unsqueeze(0).to(self.Q_current.device)
        with torch.no_grad():
            actions=self.Q_current(state)
            _, action=actions.max(1)
        return action.item()

        
        raise NotImplementedError
