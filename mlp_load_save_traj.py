import gym
import os
import numpy as np
import pickle
import gym_minigrid
from gym_minigrid import wrappers

import torch
import torch.nn as nn

import pfrl
from pfrl.agents import PPO
from pfrl.utils.batch_states import batch_states

from imitation.data.types import Trajectory


from modules.pfrl_networks import get_mlp_model


import argparse

parser = argparse.ArgumentParser(description="Full Process")

parser.add_argument(
    "--weights_path",
    "-w",
    type=str,
    default="mlp_run/50000_finish/",
    help="Path to weights",
)
parser.add_argument(
    "--env_name", "-e", type=str, default="MiniGrid-Empty-5x5-v0",
)
parser.add_argument(
    "--save_name", "-s", default="check_run", help="Save pkl file with this name",
)

parser.add_argument(
    "--num_traj", "-nt", type=int, default=50, help="How many traj to save",
)
parser.add_argument(
    "--time_limit", "-tl", type=int, default=200,
)
parser.add_argument(
    "--render", "-r", action="store_true",
)
args = parser.parse_args()

env = gym.make(args.env_name)
env = wrappers.FlatObsWrapper(env)

obs_size = env.reset().shape[0]
n_actions = env.action_space.n

model = get_mlp_model(obs_size, n_actions)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

agent = PPO(model, opt, gpu=0, recurrent=False, act_deterministically=False,)
agent.load(args.weights_path)
agent.training = False

traj_dataset = []
for traj in range(args.num_traj):
    obs_list = []
    action_list = []
    info_list = []
    obs = env.reset()
    obs_list.append(obs)
    for i in range(args.time_limit):
        action = agent._batch_act_eval(batch_states([obs], agent.device, agent.phi))
        obs, reward, done, info = env.step(action)
        action_list.append(action)
        # info_list.append({})
        obs_list.append(obs)
        if args.render:
            env.render()
        if done:
            break
    # traj_dataset.append(
    #     Trajectory(
    #         obs=np.array(obs_list),
    #         acts=np.array(action_list),
    #         # infos=np.array(info_list),
    #     )
    # )

    traj_dataset.append([np.array(obs_list), np.array(action_list)])

with open(args.save_name + ".pkl", "wb") as handle:
    pickle.dump(traj_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
