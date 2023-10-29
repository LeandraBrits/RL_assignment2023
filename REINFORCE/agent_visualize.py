# Import necessary libraries
from pathlib import Path
import gym
import pygame
import numpy as np
import cv2  # OpenCV for video creation
import os
import minihack
from pygame.locals import *

from nle import nethack
import numpy as np
import torch
from collections import deque
from frame_state import frame_state
from agent import Policy_Gradient

# REQUIREMENTS:
# Ensure you have the required libraries installed by running:
# pip install pygame opencv-python minihack

# Function to scale an observation to a new size using Pygame
def scale_observation(observation, new_size):
    """
    Scale an observation (image) to a new size using Pygame.
    Args:
        observation (pygame.Surface): The input Pygame observation.
        new_size (tuple): The new size (width, height) for scaling.
    Returns:
        pygame.Surface: The scaled observation.
    """
    return pygame.transform.scale(observation, new_size)

# Function to render the game observation
def render(obs, screen, font, text_color):
    """
    Render the game observation on the Pygame screen.
    Args:
        obs (dict): Observation dictionary containing "pixel" and "message" keys.
        screen (pygame.Surface): The Pygame screen to render on.
        font (pygame.Font): The Pygame font for rendering text.
        text_color (tuple): The color for rendering text.
    """
    img = obs["pixel"]
    msg = obs["message"]
    msg = msg[: np.where(msg == 0)[0][0]].tobytes().decode("utf-8")
    rotated_array = np.rot90(img, k=-1)

    window_size = screen.get_size()
    image_surface = pygame.surfarray.make_surface(rotated_array)
    image_surface = scale_observation(image_surface, window_size)

    screen.fill((0, 0, 0))
    screen.blit(image_surface, (0, 0))

    text_surface = font.render(msg, True, text_color)
    text_position = (window_size[0] // 2 - text_surface.get_width() // 2, window_size[1] - text_surface.get_height() - 20)
    screen.blit(text_surface, text_position)
    pygame.display.flip()

# Function to record a video of agent gameplay
def record_video(env, agent, video_filepath, pygame_frame_rate, video_frame_rate, max_timesteps):
    """
    Record a video of agent's gameplay and save it as an MP4 file.
    Args:
        env (gym.Env): The environment in which the agent plays.
        agent (object): The agent that interacts with the environment.
        video_filepath (Path): The file path where the video will be saved.
        pygame_frame_rate (int): Frame rate for rendering the video.
        video_frame_rate (int): Frame rate for the output video.
        max_timesteps (int): Maximum number of timesteps to record in the video.
    """
    frame_width = env.observation_space["pixel"].shape[1]
    frame_height = env.observation_space["pixel"].shape[0]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_filepath), fourcc, video_frame_rate, (frame_width, frame_height))

    pygame.init()
    screen = pygame.display.set_mode((frame_width, frame_height))
    font = pygame.font.Font(None, 36)
    text_color = (255, 255, 255)

    done = False
    obs = env.reset()
    clock = pygame.time.Clock()
    ################################################################################################################
    no_frames = 4
    framed_states = frame_state(no_frames)
    framed_states.reset(state)      
    ################################################################################################################
    steps = 1
    import random
    while not done and steps < max_timesteps:
        crop = stack_crop(framed_states)
        whole = stack_whole(framed_states)
        stats = stack_stats(framed_states)
        action = agent.select_action(crop, whole, stats)
        print(action)
        rand=random.randrange(0,11)
        obs, _, done, _ = env.step(rand)
        framed_states.step(obs)
        render(obs, screen, font, text_color)

        # Capture the current frame and save it to the video
        pygame.image.save(screen, "temp_frame.png")
        frame = cv2.imread("temp_frame.png")
        out.write(frame)
        
        clock.tick(pygame_frame_rate)
        steps += 1
    framed_states.reset(state)
    out.release()  # Release the video writer
    cv2.destroyAllWindows()  # Close any OpenCV windows
    os.remove("temp_frame.png")  # Remove the temporary frame file

# Function to visualize agent's gameplay and save it as a video
def visualize(env, agent, pygame_frame_rate, video_frame_rate, save_dir, max_timesteps):
    """
    Visualize agent's gameplay and save it as a video.
    Args:
        env (gym.Env): The environment in which the agent plays.
        agent (object): The agent that interacts with the environment.
        pygame_frame_rate (int): Frame rate for rendering on the pygame screen.
        video_frame_rate (int): Frame rate for the output video.
        save_dir (str): Directory where the video will be saved.
        max_timesteps (int): Maximum number of timesteps to record in the video.
    """
    os.makedirs(save_dir, exist_ok=True)
    video_filepath = Path(save_dir) / "video.mp4"

    record_video(
        env, 
        agent, 
        video_filepath, 
        pygame_frame_rate, 
        video_frame_rate,
        max_timesteps
    )

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

if __name__ == "__main__":
    # TODO: TRAINING CODE HERE ...
    ################################################################################################################
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

    env = gym.make('MiniHack-Quest-Hard-v0',
                observation_keys=("pixel","glyphs",
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
                seeds=[99]
                )
    print('yes..')
    
    
    state = env.reset()
    crop_state = torch.tensor(np.array(state["glyphs_crop"]), dtype=torch.float).unsqueeze(0)
    whole_state = torch.tensor(np.array(state["glyphs"]), dtype=torch.float).unsqueeze(0)
    stats_state = torch.tensor(state["blstats"], dtype=torch.float)
    max_steps = env._max_episode_steps
    env_action_space = env.action_space

    gamma = 1
    lr = 5e-3
    eps = 1e-6 #Adam epsilon
    decay = 1e-6#weight decay for Adam Optimizer
    no_frames = 4
    PG_agent = Policy_Gradient(crop_state, whole_state, stats_state, env.action_space, gamma, lr, eps, no_frames, decay)
    PG_agent.policy.load_state_dict(torch.load('reinf_quest_WR.pth'))
    ################################################################################################################
    # VISUALIZATION HERE ...
    #env = gym.make("MiniHack-River-v0", observation_keys=("pixel", "message"))
    
    # CAUTION: This is a random policy, so it will not perform well on the game.
    # Use your trained agent instead, which should have the `act` method.
    class RandomPolicy:
        def act(self, obs):
            return env.action_space.sample()
        
    # Visualize trained agent
    visualize(
        env, 
        PG_agent,
        pygame_frame_rate=60,
        video_frame_rate=5,
        save_dir="videos",
        max_timesteps=100
    )