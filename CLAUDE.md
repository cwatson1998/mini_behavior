# Mini-BEHAVIOR Codebase Documentation

## Note from Chris:
We did not actually use the code that Claude wrote.

## Overview
Mini-BEHAVIOR is a RL environment for household tasks built on top of Gymnasium. It supports both RGB and discrete observation spaces for training RL agents.

## Key Files and Structure

### Training Scripts
- `train_rl_agent.py` - RGB observation training using CNN policies
- `train_rl_agent_discrete.py` - Discrete observation training using MLP policies
- `manual_control.py` - Manual environment control for testing

### Core Environment Files
- `mini_behavior/minibehavior.py` - Base environment class
- `mini_behavior/grid.py` - Grid encoding and discrete observation generation
- `mini_behavior/utils/wrappers.py` - Environment wrappers
- `mini_behavior/register.py` - Environment registration

## Observation Spaces

### Discrete Observations (Default)
- **Format**: 3D array with shape `(width, height, pixel_dim)` where `pixel_dim = 32`
- **Encoding**: Each cell encodes furniture (4 values) + objects (24 values) + metadata (4 values)
- **Object mapping**: 96+ objects mapped to integers in `mini_bddl/objs.py` via `OBJECT_TO_IDX`
- **Partial obs**: Agent's limited view using `gen_obs()`
- **Full obs**: Complete grid using `gen_full_obs()` with `MiniBHFullyObsWrapper`

### RGB Observations
- **Format**: Rendered images for visualization
- **Usage**: Primarily for human interaction and visualization
- **Wrapper**: `ImgObsWrapper` converts dict observations to image-only

## Environment Configuration

### Task Registration
```python
env_name = f"MiniGrid-{task}-{room_size}x{room_size}-N2-v0"
register(
    id=env_name,
    entry_point=f'mini_behavior.envs:{task}Env',
    kwargs={"room_size": room_size, "max_steps": max_steps}
)
```

### Supported Tasks
- `PuttingAwayDishesAfterCleaning` (supports dense_reward)
- `WashingPotsAndPans` (supports dense_reward)
- Other household tasks (check `mini_behavior/envs/`)

## Wrappers

### `MiniBHFullyObsWrapper`
- **Purpose**: Enables full discrete observations instead of partial
- **Usage**: `env = MiniBHFullyObsWrapper(env)`
- **Effect**: Sets `env.use_full_obs = True`

### `ImgObsWrapper`
- **Purpose**: Converts dict observations to image-only format
- **Usage**: `env = ImgObsWrapper(env)`
- **Note**: Use for RGB training, avoid for discrete training

## RL Algorithm Support

### For RGB Observations
- **Algorithm**: PPO with `CnnPolicy`
- **Feature Extractor**: `MinigridFeaturesExtractor` (CNN-based)
- **File**: `train_rl_agent.py`

### For Discrete Observations
- **Algorithms**: PPO or DQN with `MlpPolicy`
- **Feature Extractor**: `DiscreteObsFeaturesExtractor` (MLP-based)
- **File**: `train_rl_agent_discrete.py`

## Training Commands

### RGB Training
```bash
python train_rl_agent.py --task PuttingAwayDishesAfterCleaning --room_size 10 --total_timesteps 5000000
```

### Discrete Training
```bash
python train_rl_agent_discrete.py --task PuttingAwayDishesAfterCleaning --algorithm PPO --partial_obs False
python train_rl_agent_discrete.py --task WashingPotsAndPans --algorithm DQN --dense_reward
```

## Model Saving Structure
- RGB models: `models/ppo_cnn/` or `models/ppo_cnn_partial/`
- Discrete models: `models/ppo_discrete/` or `models/dqn_discrete/`

## Development Notes

### Observation Space Handling
- Always check if observation space is `gym.spaces.Dict` vs `gym.spaces.Box`
- Dict format: `{'image': array, 'mission': str}`
- For discrete training, use the 'image' component only

### Common Pitfalls
- Don't use `ImgObsWrapper` with discrete observations
- Ensure feature extractor matches observation space (CNN for RGB, MLP for discrete)
- Use `MiniBHFullyObsWrapper` for full discrete observations

### Testing
- Use `manual_control.py` for interactive testing
- Check observation shapes before training
- Verify wrapper application order

## Dependencies
- gymnasium
- stable-baselines3
- torch
- wandb
- numpy
- mini_behavior (local package)