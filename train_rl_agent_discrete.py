import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from mini_behavior.utils.wrappers import MiniBHFullyObsWrapper
from mini_behavior.register import register
import mini_behavior
from stable_baselines3 import PPO, DQN
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback


parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True, help='name of task to train on')
parser.add_argument("--partial_obs", default=True)
parser.add_argument("--room_size", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--total_timesteps", type=int, default=5e6)
parser.add_argument("--dense_reward", action="store_true")
parser.add_argument("--algorithm", default="PPO", choices=["PPO", "DQN"], help="RL algorithm to use")
parser.add_argument("--policy_type", default="MlpPolicy", help="Policy type for discrete observations")
args = parser.parse_args()
partial_obs = args.partial_obs


class DiscreteObsFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512) -> None:
        super().__init__(observation_space, features_dim)
        
        # Handle dict observation space (includes 'image' and 'mission')
        if isinstance(observation_space, gym.spaces.Dict):
            image_space = observation_space.spaces['image']
            # Flatten the 3D discrete observation space
            n_input_features = np.prod(image_space.shape)
        else:
            # Handle box observation space directly
            n_input_features = np.prod(observation_space.shape)
        
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_input_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations) -> torch.Tensor:
        # Handle dict observations
        if isinstance(observations, dict):
            # Use only the image part for discrete observations
            observations = observations['image']
        
        return self.network(observations)


# Discrete observation policy configuration
policy_kwargs = dict(
    features_extractor_class=DiscreteObsFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=256),
)

# Env wrapping
env_name = f"MiniGrid-{args.task}-{args.room_size}x{args.room_size}-N2-v0"

print(f'register env {args.task}')

kwargs = {"room_size": args.room_size, "max_steps": args.max_steps}
if args.dense_reward:
    assert args.task in ["PuttingAwayDishesAfterCleaning", "WashingPotsAndPans"]
    kwargs["dense_reward"] = True

register(
    id=env_name,
    entry_point=f'mini_behavior.envs:{args.task}Env',
    kwargs=kwargs
)

config = {
    "policy_type": args.policy_type,
    "total_timesteps": args.total_timesteps,
    "env_name": env_name,
    "algorithm": args.algorithm,
    "observation_type": "discrete",
    "partial_obs": partial_obs,
}

print('init wandb')
run = wandb.init(
    project=f"{env_name}_discrete",
    config=config,
    sync_tensorboard=True,
    monitor_gym=False,
    save_code=True,
)

print('make env')
env = gym.make(env_name)
if not args.partial_obs:
    env = MiniBHFullyObsWrapper(env)

# Note: No ImgObsWrapper here - we want to keep discrete observations

print(f'begin training with {args.algorithm}')

# Choose algorithm based on argument
if args.algorithm == "PPO":
    model = PPO(
        config["policy_type"], 
        env, 
        n_steps=8000, 
        policy_kwargs=policy_kwargs, 
        verbose=1, 
        tensorboard_log=f"./runs/{run.id}"
    )
elif args.algorithm == "DQN":
    model = DQN(
        config["policy_type"], 
        env, 
        policy_kwargs=policy_kwargs, 
        verbose=1, 
        tensorboard_log=f"./runs/{run.id}"
    )

model.learn(config["total_timesteps"], callback=WandbCallback(model_save_path=f"models/{run.id}"))

# Save model with discrete observation identifier
model_dir = f"models/{args.algorithm.lower()}_discrete"
if not partial_obs:
    model_path = f"{model_dir}_full/{env_name}"
else:
    model_path = f"{model_dir}_partial/{env_name}"

model.save(model_path)

print(f"Model saved to: {model_path}")
run.finish()