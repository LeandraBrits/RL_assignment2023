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
def reward_manager(state, reward, count):
    #These are some messages that the agent could receive from the env when it does certain actions
    # message receieved from env:
    msg = np.array(state['message'])

    #lava message
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
    
    #lava:
    if(np.array_equiv(msg, lava_msg)):
        reward += 0.5
    
   # Message for hitting into a wall.
    wall_hit_msg=np.array([ 73, 116,  39, 115,  32,  97,  32, 119,  97, 108, 108,  46,   0,   0,   0,   0,   0,   0,\
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
    stone_walk_msg=np.array([ 73, 116,  39, 115,  32, 115, 111, 108, 105, 100,  32, 115, 116, 111, 110, 101,  46,   0,\
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

    # Reward for walking into wall
    if(np.array_equiv(msg, stone_walk_msg) or np.array_equiv(msg, wall_hit_msg)):
        reward -= 0.1#negetive reward for walking into or hitting a wall

   #walking over a wand
    wand_walk_msg=[102,  32,  45,  32,  97,  32, 114, 117, 110, 101, 100,  32, 119,  97, 110, 100,  46,   0,\
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
    # Reward for finding a wand
    if(np.array_equiv(msg, wand_walk_msg)):
        reward += 1

    #picking up a wand.
    wand_pickup_msg=[ 89, 111, 117,  32, 115, 101, 101,  32, 104, 101, 114, 101,  32,  97,  32, 114, 117, 110,\
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
   # Reward for picking up a wand:
    if(np.array_equiv(msg, wand_pickup_msg)):
        if count < 1:#so no reward for trying to pick up wand if already has one
            reward += 1
            count += 1
   #zapping without a wand.
    zap_msg=[ 89, 111, 117,  32, 100, 111, 110,  39, 116,  32, 104,  97, 118, 101,  32,  97, 110, 121,\
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
    
    #Reward for zapping without a wand.
    if(np.array_equiv(msg, zap_msg)):
        reward -= 0.2
  
    return reward, count

#functions for using the different observation states
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

#actions for agent to take
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
               seeds=[42]
              )
print('yes..')
no_frames = 4
framed_states = frame_state(no_frames)
state = env.reset()
crop_state = torch.tensor(np.array(state["glyphs_crop"]), dtype=torch.float).unsqueeze(0)
whole_state = torch.tensor(np.array(state["glyphs"]), dtype=torch.float).unsqueeze(0)
stats_state = torch.tensor(state["blstats"], dtype=torch.float)
max_steps = env._max_episode_steps
env_action_space = env.action_space

gamma = 1
lr = 5e-3
eps = 1e-6 
decay = 1e-6

#creating agent
PG_agent = Policy_Gradient(crop_state, whole_state, stats_state, env.action_space, gamma, lr, eps, no_frames, decay)

state = env.reset()
framed_states.reset(state)

#hyper parameters
num_steps = 500000#int(3e6)
ep_reward = [0.0]
ep_loss=[]
episodes = 1
steps = 0

action = None
wand_count = 0
import time
doneCounter=0
start = time.perf_counter()
for i in range(num_steps):
    
    steps+=1
    crop = stack_crop(framed_states)
    whole = stack_whole(framed_states)
    stats = stack_stats(framed_states)
    action = PG_agent.select_action(crop, whole, stats)
    state, reward, done, info = env.step(action)
    
    reward, wand_count = reward_manager(state, reward, wand_count)
    
    framed_states.step(state)

    PG_agent.rewards.append(reward)
    ep_reward[-1] += reward
    if i%1000==0:
        print('step:',i)
    if ep_reward[-1] <-100:#if rewards gets too low and episode is not ending
        done = True
        #print('over$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        doneCounter+=1
    
    if done:
        steps=0
        loss = PG_agent.train()
        ep_loss.append(loss.item())
        state = env.reset()
        wand_count=0
        framed_states.reset(state)
        ep_reward.append(0.0)
        episodes += 1
        
        
        if episodes % 100 == 0:
            print('Episode: ', episodes,'- Average Reward of last 100 episodes: ',np.mean(ep_reward[-100:]))
        #break
print('doneCounter',doneCounter)
end = time.perf_counter()
print(f"completed in {end - start:0.4f} seconds")

#saving model
torch.save(PG_agent.policy.state_dict(), "reinf_lava_NR.pth")
env.close()

#plotting results
import matplotlib.pyplot as plt
plt.figure()
plt.plot(ep_reward[:-2])
plt.title("Episode Rewards")
plt.xlabel("Episodes")
plt.ylabel("Average rewards")
plt.savefig("reinf_lava_NR_re_graph.png")
#plt.show()

plt.figure()
plt.plot(ep_loss[:-2])
plt.title("Loss curve")
plt.xlabel("Episodes")
plt.ylabel("Loss")
plt.savefig("reinf_lava_NR_loss_graph.png")
#plt.show()

np.save("reinf_lava_NR_reward_array.npy",ep_reward)
np.save("reinf_lava_NR_loss_array.npy",ep_loss)