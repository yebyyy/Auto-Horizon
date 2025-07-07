import gymnasium as gym
from gymnasium.spaces import Box  # Box is used for continuous action and observation spaces
import numpy as np
import time

from scripts.capture import get_frame_stack, GRAY
from envs.actions import do_action

class FH4Env(gym.Env):
    """
    Wrapper for Forza Horizon 4 environment.
    observation: 4 * 84 * 84 stack of frames (float32, normalized [0,1])
    actions: [steer, gas, brake], steer in [-1, 1], gas/brake in [0, 1]
    reward: +1 for every step (tbd)
    """
    metadata = {'render.modes': []} 

    def __init__(self, fps: int = 30):
        super().__init__()
        self.observation_space = Box(
            low = 0.0,
            high = 1.0,
            shape = (4, 1, 84, 84) if GRAY else (4, 3, 84, 84),  # 4 frames of size 84x84
            dtype= np.float32
        )
        self.action_space = Box(
            low = np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high = np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
        self._dt = 1.0 / fps  # step interval ~0.0333 seconds

    def reset(self, seed=None, options=None):
        """
        Before reset, load solo road race, position the car at the start grid.
        """
        obs = get_frame_stack()
        info = {}
        return obs, info
    
    def step(self, action):
        start = time.perf_counter()
        do_action(action)

        obs = get_frame_stack()
        reward = 1.0           # TODO: implement reward
        terminated = False     # TODO: implement termination conditions
        truncated = False  # truncated means the episode ended due to time limit, thus not using it here
        info = {}

        elapsed = time.perf_counter() - start
        if elapsed < self._dt:
            time.sleep(self._dt - elapsed)

        return obs, reward, terminated, truncated, info