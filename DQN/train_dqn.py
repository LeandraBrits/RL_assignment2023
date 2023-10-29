from dqn_helper_files.agent import DQNAgent
from dqn_helper_files.replay_buffer import ReplayBuffer
from dqn_helper_files.wrappers import *
from dqn_helper_files.explore import *

import random
import torch 
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.wrappers.monitoring import video_recorder as VideoRecorder
from nle import nethack
import minihack

from tempfile import TemporaryFile
import time

if __name__ == '__main__':
    start = time.perf_counter()
    outfile = TemporaryFile()

    # actions chosen from minihack documentation
    # we have chosen actions that we think will aid our agent through the quest hard environment

    MOVE_ACTIONS = tuple(nethack.CompassDirection)
    
    NAVIGATE_ACTIONS = MOVE_ACTIONS + (
    nethack.Command.OPEN,  
    nethack.Command.PICKUP, 
    nethack.Command.WEAR,   
    nethack.Command.WIELD,  
    nethack.Command.QUAFF,
    nethack.Command.INVOKE,
    nethack.Command.ZAP,
    nethack.Command.SWAP,
    nethack.Command.FIRE,
    nethack.Command.KICK,
    nethack.Command.SEARCH
)
    
    # implementing a custom reward manager to help guide our agent further

    reward_manager = RewardManager()
    
    # reward killing the monster
    reward_manager.add_kill_event("minotaur", reward=1, terminal_required=False)

    # reward wielding items that award levitation 
    reward_manager.add_wield_event("levitation", reward=1, terminal_required=False)
    
    # reward wielding items that award freezing stuff 
    reward_manager.add_wield_event("cold", reward=1, terminal_required=False)
    
    # reward wielding wand of death
    reward_manager.add_wield_event("death", reward=1, terminal_required=False)
    
    # a message that occurs when an agent walks into a wall; negatively reward walking into the wall
    reward_manager.add_message_event("It's solid stone.", reward=-0.75, terminal_required=False, repeatable=True)

    # reward the agent for walking around if it hasnt reached its goal
    reward_manager.add_event(ExploreEvent(0.5, True, True, False))
    
    hyper_params = {
        "seed": 42,  # which seed to use
        "env-name": "MiniHack-Quest-Hard-v0",  # name of the game
        "replay-buffer-size": int(5e3),  # replay buffer size
        "learning-rate": 1e-3,  # learning rate for Adam optimizer
        "discount-factor": 0.99,  # discount factor
        "num-steps": 500000,  # total number of steps to run the environment for
        "batch-size": 128,  # number of transitions to optimize at the same time
        "learning-starts": 1000,  # number of steps before learning starts
        "learning-freq": 2,  # number of iterations between every optimization step
        "use-double-dqn": True,  # use double deep Q-learning
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": 1.0,  # e-greedy start threshold
        "eps-end": 0.01,  # e-greedy end threshold
        "eps-fraction": 0.3,  # fraction of num-steps
        "print-freq": 25, # number of iterations between each print out
    }

                    
    env = gym.make(
        hyper_params["env-name"],

        # observation_keys define what observations are received back from the environment
        observation_keys=("glyphs_crop", "chars", "colors", "pixel", "message", "blstats", "pixel_crop"),
        actions=NAVIGATE_ACTIONS,
  
        reward_win=1, # the reward received upon successfully completing an episode.
        reward_lose=0, # the reward received upon death or aborting
        reward_manager=reward_manager
    )

    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])
    
    # need these wrappers to format the environment correctly to work with our DQN
    
    env.seed(hyper_params["seed"])
    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = FrameStack(env, 4)
    
    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])

    agent = DQNAgent(
        env.observation_space,
        env.action_space,
        replay_buffer,
        use_double_dqn=hyper_params["use-double-dqn"],
        lr = hyper_params["learning-rate"], 
        batch_size = hyper_params["batch-size"], 
        gamma = hyper_params["discount-factor"] 

    )
    
    # the following code is slightly adapted from code provided for the DQN lab completed earlier this semester.

    eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])
    episode_rewards = [0.0] 
    episode_loss = [0.0]
    state = env.reset()

    for t in range(hyper_params["num-steps"]):
        if t%10000==0:
            print(t)
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * (
            hyper_params["eps-end"] - hyper_params["eps-start"]
        )
        sample = random.random()

        #  select random action if sample is less equal than eps_threshold
    
        if sample>eps_threshold:
            action = agent.act(state)
            
        else:
            action = env.action_space.sample()

        next_state, reward, done, info =env.step(action) # take step in env

        # adding state, action, reward, next_state, float(done) to reply memory
        agent.replay_buffer.add(state, action, reward, next_state, float(done))
        state = next_state

        episode_rewards[-1] += reward # add reward to episode_reward

        if done:
            state = env.reset()
            episode_rewards.append(0.0)

        if (
            t > hyper_params["learning-starts"]
            and t % hyper_params["learning-freq"] == 0
        ):
            loss=agent.optimise_td_loss()
            
            episode_loss.append(loss)

        if (
            t > hyper_params["learning-starts"]
            and t % hyper_params["target-update-freq"] == 0
        ):
            agent.update_target_network()

        num_episodes = len(episode_rewards)

        if (
            done
            and hyper_params["print-freq"] is not None
            and len(episode_rewards) % hyper_params["print-freq"] == 0
        ):
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")

    end = time.perf_counter()
    print(f"completed in {end - start:0.4f} seconds")
    torch.save(agent.Q_current.state_dict(), "dqn_quest_WR.pth")

    # plots of the episodes vs loss and rewards

    plt.figure()
    plt.plot(episode_rewards[:-2])
    plt.title("Episode Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Average rewards")
    plt.savefig("dqn_quest_WR_re_graph.png")

    plt.figure()
    plt.plot(episode_loss[:-2])
    plt.title("Loss curve")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.savefig("dqn_quest_WR_loss_graph.png")

    np.save("quest_WR_reward_array.npy",episode_rewards)
    np.save("quest_WR_loss_array.npy",episode_loss)
    agent.save("quest_WR_dqn")