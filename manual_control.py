#!/usr/bin/env python3

import argparse
import gymnasium as gym # added by chris
from minigrid.wrappers import *
from mini_behavior.window import Window
from mini_behavior.utils.save import get_step, save_demo
from mini_behavior.grid import GridDimension
from mini_behavior.utils.wrappers import MiniBHCumulativeFovWrapper
import numpy as np

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32
show_furniture = False


def redraw(img):
    if not args.agent_view:
        # This seems suspicious
        env.get_wrapper_attr('set_render_mode')('rgb_array')
        img = env.render()

    window.no_closeup()
    window.set_inventory(env.unwrapped)
    window.show_img(img)


def render_furniture():
    global show_furniture
    show_furniture = not show_furniture

    if show_furniture:
        img = np.copy(env.furniture_view)

        # i, j = env.agent.cur_pos
        i, j = env.get_wrapper_attr('agent_pos')
        ymin = j * TILE_PIXELS
        ymax = (j + 1) * TILE_PIXELS
        xmin = i * TILE_PIXELS
        xmax = (i + 1) * TILE_PIXELS

        img[ymin:ymax, xmin:xmax, :] = GridDimension.render_agent(
            img[ymin:ymax, xmin:xmax, :], env.get_wrapper_attr('agent_dir'))
        img = env.render_furniture_states(img)

        window.show_img(img)
    else:
        obs = env.get_wrapper_attr('gen_obs')()
        redraw(obs)


def show_states():
    imgs = env.render_states()
    window.show_closeup(imgs)


def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission') or env.has_wrapper_attr('mission'):
        mission = env.get_wrapper_attr('mission')
        print('Mission: %s' % mission)
        window.set_caption(mission)

    redraw(obs)


def load():
    if args.seed != -1:
        env.seed(args.seed)

    env.reset()
    obs = env.load_state(args.load)

    if hasattr(env, 'mission') or env.has_wrapper_attr('mission'):
        mission = env.get_wrapper_attr('mission')
        print('Mission: %s' % mission)
        window.set_caption(mission)

    redraw(obs)


def step(action):
    prev_obs = env.get_wrapper_attr('gen_obs')()
    obs, reward, done, truncated, info = env.step(action)

    step_count = env.get_wrapper_attr('step_count')
    print('step=%s, reward=%.2f' % (step_count, reward))

    if args.save:
        all_steps[step_count] = (prev_obs, action)

    if done:
        print('done!')
        if args.save:
            episode = env.get_wrapper_attr('episode')
            save_demo(all_steps, args.env, episode)
        reset()
    else:
        redraw(obs)


def switch_dim(dim):
    env.switch_dim(dim)
    render_dim = env.get_wrapper_attr('render_dim')
    print(f'switching to dim: {render_dim}')
    obs = env.get_wrapper_attr('gen_obs')()
    redraw(obs)


def key_handler_cartesian(event):
    print('pressed', event.key)
    if event.key == 'escape':
        window.close()
        return
    if event.key == 'backspace':
        reset()
        return
    if event.key == 'left':
        actions = env.get_wrapper_attr('actions')
        step(actions.left)
        return
    if event.key == 'right':
        actions = env.get_wrapper_attr('actions')
        step(actions.right)
        return
    if event.key == 'up':
        actions = env.get_wrapper_attr('actions')
        step(actions.forward)
        return
    # Spacebar
    if event.key == ' ':
        render_furniture()
        return
    if event.key == 'pageup':
        step('choose')
        return
    if event.key == 'enter':
        env.save_state()
        return
    if event.key == 'pagedown':
        show_states()
        return
    if event.key == '0':
        switch_dim(None)
        return
    if event.key == '1':
        switch_dim(0)
        return
    if event.key == '2':
        switch_dim(1)
        return
    if event.key == '3':
        switch_dim(2)
        return

def key_handler_primitive(event):
    print('pressed', event.key)
    if event.key == 'escape':
        window.close()
        return
    if event.key == 'left':
        actions = env.get_wrapper_attr('actions')
        step(actions.left)
        return
    if event.key == 'right':
        actions = env.get_wrapper_attr('actions')
        step(actions.right)
        return
    if event.key == 'up':
        actions = env.get_wrapper_attr('actions')
        step(actions.forward)
        return
    if event.key == '0':
        actions = env.get_wrapper_attr('actions')
        step(actions.pickup_0)
        return
    if event.key == '1':
        actions = env.get_wrapper_attr('actions')
        step(actions.pickup_1)
        return
    if event.key == '2':
        actions = env.get_wrapper_attr('actions')
        step(actions.pickup_2)
        return
    if event.key == '3':
        actions = env.get_wrapper_attr('actions')
        step(actions.drop_0)
        return
    if event.key == '4':
        actions = env.get_wrapper_attr('actions')
        step(actions.drop_1)
        return
    if event.key == '5':
        actions = env.get_wrapper_attr('actions')
        step(actions.drop_2)
        return
    if event.key == 't':
        actions = env.get_wrapper_attr('actions')
        step(actions.toggle)
        return
    if event.key == 'o':
        actions = env.get_wrapper_attr('actions')
        step(actions.open)
        return
    if event.key == 'c':
        actions = env.get_wrapper_attr('actions')
        step(actions.close)
        return
    if event.key == 'k':
        actions = env.get_wrapper_attr('actions')
        step(actions.cook)
        return
    if event.key == 's':
        actions = env.get_wrapper_attr('actions')
        step(actions.slice)
        return
    if event.key == 'i':
        actions = env.get_wrapper_attr('actions')
        step(actions.drop_in)
        return
    if event.key == 'pagedown':
        show_states()
        return
    else:
        print("Illegal key. No-op")
        return
    

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-InstallingAPrinter-8x8-N2-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view). Needs fixing.",
    action='store_true'
)
parser.add_argument(
    '--cumulative_fov',
    default=False,
    help="draw everything the agent has seen (like having clairvoyant memory but limited view). Do not combine with agent_view.",
    action='store_true'
)

# NEW
parser.add_argument(
    "--save",
    default=False,
    help="whether or not to save the demo_16"
)
# NEW
parser.add_argument(
    "--load",
    default=None,
    help="path to load state from"
)

args = parser.parse_args()

env = gym.make(args.env)
# Chris added the .unwrapped everywhere it appears in this file.
env.get_wrapper_attr('teleop_mode')()

if args.save:
    # We do not support save for cartesian action space
    mode = env.get_wrapper_attr('mode')
    assert mode == "primitive"

all_steps = {}

if args.agent_view:
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
if args.cumulative_fov:
    env = MiniBHCumulativeFovWrapper(env)
print(type(env))
print(type(env.env))
print(type(env.env.env))
window = Window('mini_behavior - ' + args.env)
mode = env.get_wrapper_attr('mode')
if mode == "cartesian":
    window.reg_key_handler(key_handler_cartesian)
elif mode == "primitive":
    window.reg_key_handler(key_handler_primitive)

if args.load is None:
    reset()
else:
    load()

# Blocking event loop
window.show(block=True)
