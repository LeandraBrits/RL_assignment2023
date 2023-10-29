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
    # from minihack documentation
    # look into adding different navigate_actions from the docs; our choice of actions can drastically affect our agent performance

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
    
    
    
    hyper_params = {
        "seed": 42,  # which seed to use
        "env-name": "MiniHack-LavaCross-Full-v0",  # name of the game
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

    # env = gym.make('MiniHack-LavaCross-Full-v0',
                   
                    # observation_keys = ['pixel_crop'], # parameter used to specify the observation space in any MiniHack environment
                    # # different observation keys can make environments significantly easier or harder
                    # # why using crop? According to docs: Cropped observations can facilitate the learning, as the egocentric input makes representation learning easier.
                    
                    # # reward function configuration

                    # # this is currently the default configuration from the minihack docs. We can look into changing this and observe performance
                    # # https://minihack.readthedocs.io/en/latest/getting-started/reward.html#default-configuration
                    # # LOOK INTO CUSTOM REWARD MANAGER
                    
                    # reward_win=1, # the reward received upon successfully completing an episode.
                    # reward_lose=0, # the reward received upon death or aborting
                    # penalty_step=-0.1, # constant applied to amount of frozen steps
                    # penalty_time=0, # constant applied to amount of non-frozen steps
            
                    # seeds = [0],
                    # actions = NAVIGATE_ACTIONS)
                    
    env = gym.make(
        hyper_params["env-name"],
        observation_keys=("glyphs_crop", "chars", "colors", "pixel", "message", "blstats", "pixel_crop"),
        actions=NAVIGATE_ACTIONS,

        # this is currently the default configuration from the minihack docs. We can look into changing this and observe performance
        # https://minihack.readthedocs.io/en/latest/getting-started/reward.html#default-configuration
        # LOOK INTO CUSTOM REWARD MANAGER
        
        reward_win=1, # the reward received upon successfully completing an episode.
        reward_lose=0, # the reward received upon death or aborting
        # penalty_step=-0.1, # constant applied to amount of frozen steps
        # penalty_time=0, # constant applied to amount of non-frozen steps
        #reward_manager=reward_manager
    )



    # torch.manual_seed(hyper_params["seed"])
    # torch.cuda.manual_seed_all(hyper_params["seed"])
    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])
    
    # need these wrappers or you'll get "AssertionError: observation_space must be of type Box"
    # look into why we need that assertion?
    
    env.seed(hyper_params["seed"])
  #  env = RenderRGB(env, "pixel_crop")
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
    #agent.Q_current.load_state_dict(torch.load('dqn_Room-Monster.pth'))
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
        # TODO
        #  select random action if sample is less equal than eps_threshold
        # take step in env
        # add state, action, reward, next_state, float(done) to reply memory - cast done to float
        # add reward to episode_reward
        if sample>eps_threshold:
            action = agent.act(state)
            
        else:
            action = env.action_space.sample()

        next_state, reward, done, info =env.step(action)
        agent.replay_buffer.add(state, action, reward, next_state, float(done))
        state = next_state

        episode_rewards[-1] += reward

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
            # print("********************************************************")
            # print("steps: {}".format(t))
            # print("episodes: {}".format(num_episodes))
            # print("mean 100 episode reward: {}".format(mean_100ep_reward))
            # print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            # print("********************************************************")

    end = time.perf_counter()
    print(f"completed in {end - start:0.4f} seconds")
    torch.save(agent.Q_current.state_dict(), "dqn_lava_NR.pth")

 

    plt.figure()
    plt.plot(episode_rewards[:-2])
    plt.title("Episode Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Average rewards")
    plt.savefig("dqn_lava_NR_re_graph.png")
    #plt.show()

    plt.figure()
    plt.plot(episode_loss[:-2])
    plt.title("Loss curve")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.savefig("dqn_lava_NR_loss_graph.png")
    #plt.show()
    np.save("lava_NR_reward_array.npy",episode_rewards)
    np.save("lava_NR_loss_array.npy",episode_loss)
    agent.save("lava_NR_dqn")
