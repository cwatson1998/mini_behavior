import gymnasium as gym
from gymnasium import spaces
from mini_bddl import OBJECT_TO_IDX
from typing import Any, Dict, Optional, Tuple
import numpy as np


class MiniBHFullyObsWrapper(gym.core.ObservationWrapper):
	"""
	Fully observable gridworld using a compact grid encoding
	MiniBH encoding is different from MiniGrid encoding
	"""

	def __init__(self, env):
		super().__init__(env)

		self.unwrapped.use_full_obs = True
		self.observation_space.spaces["image"] = spaces.Box(
			low=0,
			high=255,
			shape=(self.env.width, self.env.height, env.grid.pixel_dim),  # number of cells
			dtype='uint8'
		)

	def observation(self, obs):
		return obs
	
"""This is super hacky because the actual core behavior is in MiniBehaviorGrid"""
class MiniBHCumulativeFovWrapper(gym.core.Wrapper):
	
	def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
		# TODO: directly accessing unwrapped is bad style.
		self.unwrapped.cumulative_highlight_mask = np.zeros(shape=(self.get_wrapper_attr("width"), self.get_wrapper_attr("height")), dtype=bool)
		return super().reset(seed=seed, options=options)
	
	
