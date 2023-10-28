import gym
import minihack
from nle import nethack
from gym import spaces
import torch.nn as nn
import numpy as np
import torch
from collections import deque
from collections import Counter
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import collections
import torch.nn.functional as F
from minihack import RewardManager
from frame_state import frame_state
from model import combined
from agent import Policy_Gradient

'''
This function contains multiple functions that change the reward earned at a timestep.
These functions were created to try help the agent learn in sparse reward environments.
'''
def reward_manager(state, last_action, action, reward, count, visited_states_map, dist, min_dist, found):

    # Exploration reward for unvisited states:
    #  if the goal state has not been found yet, add an exploration reward proportional
    #  to the number of times that a state has been visited.
    #  Created to assist with Maze Exploration.
    '''if not found:
        beta = 0.01
        visited_states_map[state['blstats'][1]][state['blstats'][0]] += 1
        occurence = visited_states_map[state['blstats'][1]][state['blstats'][0]]
        reward += beta*(1/np.sqrt(occurence))'''


    # Closer distance to goal reward:
    #  Give the agent a reward if the distance in current state is closer to the goal
    #  state than we have previously found.
    #  Created to assist with Maze Exploration.
    '''if dist < min_dist:
        reward+=0.01
        min_dist = dist'''

    # Message for successfully solidifying the lava.
    #  Created to help with the LavaCross task.
    lava_msg = np.array([ 84, 104, 101,  32, 108,  97, 118,  97,  32,  99, 111, 111, 108, \
       115,  32,  97, 110, 100,  32, 115, 111, 108, 105, 100, 105, 102,\
       105, 101, 115,  46,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
         0,   0,   0,   0,   0,   0,   0,   0,   0])

    # Message for killing a newt
    newt_msg = np.array([ 89, 111, 117,  32, 107, 105, 108, 108,  32, 116, 104, 101,  32, 110, 101 ,119, 116,  33,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0])
    # Message for killing a lichen.
    lichen_msg = np.array([ 89, 111, 117,  32, 107, 105, 108, 108,  32, 116, 104, 101,  32, 108, 105,  99, 104, 101,\
 110,  33,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0])
   # Message for hitting into a wall.
    wall_msg=np.array([ 73, 116,  39, 115,  32,  97,  32, 119,  97, 108, 108,  46,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0])
   #Message for walking into stone wall.
    stone_msg=np.array([ 73, 116,  39, 115,  32, 115, 111, 108, 105, 100,  32, 115, 116, 111, 110, 101,  46,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0])

   #Message for successfully walking over a wand up a wand.
    wand_msg=[102,  32,  45,  32,  97,  32, 114, 117, 110, 101, 100,  32, 119,  97, 110, 100,  46,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0]
    #message for picking up a wand.
    wand_find_msg=[ 89, 111, 117,  32, 115, 101, 101,  32, 104, 101, 114, 101,  32,  97,  32, 114, 117, 110,\
 101, 100,  32, 119,  97, 110, 100,  46,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0]
   #Message for zapping without a wand.
    no_zap_msg=[ 89, 111, 117,  32, 100, 111, 110,  39, 116,  32, 104,  97, 118, 101,  32,  97, 110, 121,\
 116, 104, 105, 110, 103,  32, 116, 111,  32, 122,  97, 112,  46,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\
   0,   0,   0,   0]
    # Current State Message:
    msg = np.array(state['message'])
    # Reward for burning the lava:
    if(np.array_equiv(msg, lava_msg)):
        reward += 0.5
        count +=1
    # Reward for killing a newt:
    if(np.array_equiv(msg, newt_msg)):
        reward += 0.5
    # Reward for killing a lichen:
    if(np.array_equiv(msg, lichen_msg)):
        reward += 0.5
    # Reward for finding a wand
    if(np.array_equiv(msg, wand_msg)):
        reward += 1
    # reward for picking up a wand:
    if(np.array_equiv(msg, wand_find_msg)):
        if count > 1:
            reward = reward
        else:
            reward += 1
            count += 1
    # Reward for walking into wall
    if(np.array_equiv(msg, stone_msg) or np.array_equiv(msg, wall_msg)):
        reward -= 0.1

    return reward, count, min_dist, visited_states_map

'''
The below four functions are used to take the three different representations of the
observation space and stack the previous 3 timestep observation onto the current observation
at time t.
'''
def stack_crop(framed_states):
    crop_state = deque([], maxlen = len(framed_states.frames))
    for i in reversed(range(len(framed_states.frames))):
        crop_state.append((framed_states.frames[i]["glyphs_crop"]-framed_states.frames[i]["glyphs_crop"].mean())/(framed_states.frames[i]["glyphs_crop"].std()+1e-8))
    crop_state = torch.tensor(np.array(crop_state), dtype=torch.float).unsqueeze(0)
    return crop_state

def stack_whole(framed_states):
    whole_state = deque([], maxlen = len(framed_states.frames))
    for i in reversed(range(len(framed_states.frames))):
        whole_state.append((framed_states.frames[i]["glyphs"]-framed_states.frames[i]["glyphs"].mean())/(framed_states.frames[i]["glyphs"].std()+1e-8))
    whole_state = torch.tensor(np.array(whole_state), dtype=torch.float).unsqueeze(0)
    return whole_state

def stack_stats(framed_states):
    stats_state = torch.tensor(framed_states.frames[-1]["blstats"], dtype=torch.float).unsqueeze(0)
    stats_state = ((stats_state-stats_state.mean())/(stats_state.std()+1e-8))
    return stats_state

'''
The dist_ function calculated Euclidean distance between the agent and the goal state
using coordinates.
'''
def dist_(x1,y1,x2,y2):
    distance = abs((y2-y1)/(x2-x1))
    return distance

'''
This function backtests the agent on an environment to analyse testing performance
of the trained model.
'''
def backtest_PG(agent, env):
    state = env.reset()
    framed_states.reset(state)
    done = False
    env.render()
    i=0
    while not done:
        crop = stack_crop(framed_states)
        whole = stack_whole(framed_states)
        stats = stack_stats(framed_states)
        action = agent.select_action(crop, whole, stats)
        print('\n selected action: ',action)
        state, reward, done, info = env.step(action)
        framed_states.step(state)
        if i < 20:
            env.render()

        i+=1
        if i > 50:#done:
            break

def backtest_PPO(agent):
    state = env.reset()
    framed_states.reset(state)
    done = False
    env.render()
    i=0
    while not done:
        crop = stack_crop(framed_states)
        whole = stack_whole(framed_states)
        stats = stack_stats(framed_states)
        action, _, _ = agent.select_action(crop, whole, stats)
        print('\n selected action: ',action)
        state, reward, done, info = env.step(action)
        framed_states.step(state)
        #if i % 20 == 0:
        env.render()

        i+=1
        if done:
            break

ACTIONS = (
    nethack.CompassDirection.N, # 0
    nethack.CompassDirection.E, # 1
    nethack.CompassDirection.S, # 2
    nethack.CompassDirection.W, # 3
    nethack.Command.ZAP, # 4
    nethack.Command.FIRE, # 5 - f , nethack.Command.RUSH, - g
    nethack.Command.PICKUP, # 6
    nethack.Command.KICK,
    nethack.Command.WIELD,
    nethack.Command.WEAR,
    nethack.Command.QUAFF
)

env = gym.make('MiniHack-River-Lava-v0',
               observation_keys=("glyphs",
                                 "glyphs_crop",
                                 "blstats",
                                 "message",
                                 "inv_strs",
                                ),
               penalty_time=-0.001,
               penalty_step=-0.01, #non-valid step
               reward_lose=-5,
               reward_win=5,
               penalty_mode="constant",
               actions=ACTIONS,
               seeds=[99] #wand - 8, 9, 13
               #reward_manager=reward_gen,
              )
print('yes..')
no_frames = 4
framed_states = frame_state(no_frames)
state = env.reset()
crop_state = torch.tensor(np.array(state["glyphs_crop"]), dtype=torch.float).unsqueeze(0)
whole_state = torch.tensor(np.array(state["glyphs"]), dtype=torch.float).unsqueeze(0)
stats_state = torch.tensor(state["blstats"], dtype=torch.float)
visited_states = collections.deque([], maxlen=env._max_episode_steps)
max_steps = env._max_episode_steps
visited_states_map = np.zeros((21,79))
env_action_space = env.action_space

gamma = 1
lr = 1e-3
eps = 1e-6 #Adam epsilon
decay = 1e-6 #weight decay for Adam Optimizer

PG_agent = Policy_Gradient(crop_state, whole_state, stats_state, env.action_space, gamma, lr, eps, no_frames, decay)
#PG_agent.policy.load_state_dict(torch.load('maze_model.pth'))
#tb = SummaryWriter(comment='Combined_Model - PG '+str(env))
state = env.reset()
framed_states.reset(state)
visited_states_map = np.zeros((21,79)) # a map of counts for each state
visited_states_map[state['blstats'][1]][state['blstats'][0]] +=1
min_dist, dist = 0, 0
num_steps = 20000#int(3e6)
ep_reward = [0.0]

PG_agent.policy.load_state_dict(torch.load('River-Lava_negetive_rewards_done_limit_RAdam_700k-1200k.pth'))

#from gym.wrappers.monitoring.video_recorder import VideoRecorder
####################################
from PIL import Image
#import gym
#import minihack
from datetime import datetime

def save_gif(gif,path):
    '''
    Args:
        gif: a list of image objects
        path: the path to save the gif to
    '''
    path=path+'.gif'
    gif[0].save(path, save_all=True,optimize=False, append_images=gif[1:], loop=0)
    print("Saved Video")

def frames_to_gif(frames):
    '''
    Converts a list of pixel value arrays to a gif list of images
    '''
    gif = []
    for image in frames:
        gif.append(Image.fromarray(image, "RGB"))
    return gif


def generate_video(model,title=None,path='../videos/'):
    '''
    Generates a gif for a model, saves it
    Args:
        model - a class with a predict method that takes in a state observation and returns an action
        title - the title of the gif 
        the path to save the gif to
    Ret:
        None
    '''
    frames = []
    env = gym.make('MiniHack-River-Lava-v0')
    done = False
    obs = env.reset()
    while not done:
        del obs['pixel']
        action = model.select_action(crop, whole, stats)
        obs, reward, done, info = env.step(action)
        print(obs)
        frames.append(obs["pixel"])
        

    if title is None:
        title = datetime.now().strftime("%d-%m-%Y_%H:%M:%S") 

    gif = frames_to_gif(frames)
    save_gif(gif,path+title)

####################################

for i in range(0):
        generate_video(PG_agent,title='PPO_GIFS/PPO_GIF_{}'.format(i))

#video=VideoRecorder(env,path="rlvid.mp4")
num_steps = 10
# Your agent training loop
state = env.reset()
frames = []
for i in range(num_steps):
    print(i)
    done = False
    counter=0
    #print(counter)
    counter+=1
    crop = stack_crop(framed_states)
    whole = stack_whole(framed_states)
    stats = stack_stats(framed_states)
    action = PG_agent.select_action(crop, whole, stats)
    next_state,reward,done,info=env.step(action)
    #env.render()
    #print(next_state['glyphs'])
    frames.append(next_state['glyphs'])
    
    #video.capture_frame()
    ep_reward[-1]+=reward
    if done:
        ep_reward.append(0.0)
        state=env.reset()
#video.close()
gif = frames_to_gif(frames)
save_gif(gif,'testvid')
env.close()


