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
    # if not found:
    #     beta = 0.01
    #     visited_states_map[state['blstats'][1]][state['blstats'][0]] += 1
    #     occurence = visited_states_map[state['blstats'][1]][state['blstats'][0]]
    #     reward += beta*(1/np.sqrt(occurence))


    # # Closer distance to goal reward:
    # #  Give the agent a reward if the distance in current state is closer to the goal
    # #  state than we have previously found.
    # #  Created to assist with Maze Exploration.
    # if dist < min_dist:
    #     reward+=0.01
    #     min_dist = dist

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
    # Reward for picking up a wand:
    if(np.array_equiv(msg, wand_find_msg)):
        if count > 1:
            reward = reward
        else:
            reward += 1
            count += 1
    # Reward for walking into wall
    if(np.array_equiv(msg, stone_msg) or np.array_equiv(msg, wall_msg)):
        #print('walk into wall')
        reward -= 0.1
    #Reward for zapping without a wand.
    if(np.array_equiv(msg, no_zap_msg)):
        reward -= 0.2

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

env = gym.make('MiniHack-MazeWalk-15x15-v0',
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
                #wand - 8, 9, 13
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
lr = 5e-3
eps = 1e-6 #Adam epsilon
decay = 1e-6#weight decay for Adam Optimizer

PG_agent = Policy_Gradient(crop_state, whole_state, stats_state, env.action_space, gamma, lr, eps, no_frames, decay)
#PG_agent.policy.load_state_dict(torch.load('River-Lava_negetive_rewards_done_limit_RAdam.pth'))
#tb = SummaryWriter(comment='Combined_Model - PG '+str(env))
state = env.reset()
framed_states.reset(state)
visited_states_map = np.zeros((21,79)) # a map of counts for each state
visited_states_map[state['blstats'][1]][state['blstats'][0]] +=1
min_dist, dist = 0, 0
num_steps = 200000#int(3e6)
ep_reward = [0.0]
ep_loss=[]
episodes = 1
steps = 0
last_action = None
action = None
lava_count = 0
found = False
import time
doneCounter=0
start = time.perf_counter()
for i in range(num_steps):
    
    steps+=1
    crop = stack_crop(framed_states)
    whole = stack_whole(framed_states)
    stats = stack_stats(framed_states)
    action = PG_agent.select_action(crop, whole, stats)
    #print('action',action)
    state, reward, done, info = env.step(action)
    #print('state',state)
    #print('reward',reward)
    #print('done',done)
    #print('info',info)
    if reward > 4.9:
        found = True
    
    #reward, lava_count, min_dist, visited_states_map = reward_manager(state, last_action, action, reward, lava_count, visited_states_map, dist, min_dist, found)
    last_action = action
    framed_states.step(state)
    #tb.add_scalar('reward_per_step', reward, i)

    PG_agent.rewards.append(reward)
    ep_reward[-1] += reward
    #if done:
        #print('done---------------------------------------------------------------------')
    if i%1000==0:
        print('step:',i)
        #if len(ep_reward)>1:
        #print('ep',episodes,': ',ep_reward[-1])
    if ep_reward[-1] <-100:
        done = True
        print('over$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        doneCounter+=1
    
    if done:
        #tb.add_scalar('num_steps_per_episode', steps, episodes)
        steps=0
        loss = PG_agent.train()
        #print(loss)
        #print(loss.item())
        ep_loss.append(loss.item())
        state = env.reset()
        lava_count=0
        framed_states.reset(state)
        visited_states_map = np.zeros((21,79))
        #tb.add_scalar('reward', ep_reward[-1], episodes)
        #print('ep',episodes,': ',ep_reward[-1])
        ep_reward.append(0.0)
        #tb.add_scalar('loss', loss, episodes)
        episodes += 1
        
        
        if episodes % 100 == 0:
            print('Episode: ', episodes,'- Av Reward of last 100 episodes: ',np.mean(ep_reward[-100:]))
        #break
print('doneCounter',doneCounter)
end = time.perf_counter()
print(f"completed in {end - start:0.4f} seconds")
#torch.save(PG_agent, "Agent_dicts/PG/PG_state_dict - "+str(env))

save_dir = "Agent_dicts/PG"
import os
os.makedirs(save_dir, exist_ok=True)
#PG_agent.policy_network.load_state_dict(torch.load('maze_model.pth'))



# from gym.wrappers.monitoring.video_recorder import VideoRecorder

# video=VideoRecorder(env,path="rlvid.mp4")

# # Your agent training loop
# for episode in range(0):
#     state = env.reset()
#     done = False
#     counter=0
#     while True and counter<10000:
#         #print(counter)
#         counter+=1
#         crop = stack_crop(framed_states)
#         whole = stack_whole(framed_states)
#         stats = stack_stats(framed_states)
#         action = PG_agent.select_action(crop, whole, stats)
        
#         next_state,reward,done,info=env.step(action)
#         #env.render()
#         #video.capture_frame()
#         if done:
#             print("done-------------------------------------------------------------------------")
#             break
#         state=next_state
# video.close()

torch.save(PG_agent.policy.state_dict(), "reinf_maze_NR.pth")
env.close()

import matplotlib.pyplot as plt
plt.figure()
plt.plot(ep_reward[:-2])
plt.title("Episode Rewards")
plt.xlabel("Episodes")
plt.ylabel("Average rewards")
plt.savefig("reinf_maze_NR_re_graph.png")
#plt.show()

plt.figure()
plt.plot(ep_loss[:-2])
plt.title("Loss curve")
plt.xlabel("Episodes")
plt.ylabel("Loss")
plt.savefig("reinf_maze_NR_loss_graph.png")
#plt.show()

np.save("reinf_maze_NR_reward_array.npy",ep_reward)
np.save("reinf_maze_NR_loss_array.npy",ep_loss)