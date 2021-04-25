import gym
import os
import numpy as np
import pickle
import gym_minigrid
from gym_minigrid import wrappers
import logging

import torch
import torch.nn as nn

import pfrl
from pfrl import experiments, utils
from pfrl.agents import PPO
from pfrl.policies import SoftmaxCategoricalHead
from pfrl.utils.batch_states import batch_states

env_name = "MiniGrid-Empty-5x5-v0"

env = gym.make(env_name)
env = wrappers.FlatObsWrapper(env)


obs_size = env.reset().shape[0]
n_actions = env.action_space.n



model = nn.Sequential(
    nn.Linear(obs_size, 64),
    nn.ReLU(),
    pfrl.nn.Branched(
        # action branch
        nn.Sequential(nn.Linear(64, n_actions), SoftmaxCategoricalHead(),),
        # value branch
        nn.Linear(64, 1),
    ),
)


opt = torch.optim.Adam(model.parameters(), lr=3e-4)


agent = PPO(
    model,
    opt,
    gpu=0,
    # update_interval=,
    minibatch_size=64,
    epochs=10,
    clip_eps=0.1,
    clip_eps_vf=None,
    standardize_advantages=True,
    entropy_coef=1e-2,
    recurrent=False,
    max_grad_norm=0.5,
    act_deterministically=True
)

agent.load("mlp_run/50000_finish/")
agent.training = False

for traj in range(30):
    obs = env.reset()
    for i in range(500):
        action = agent._batch_act_eval(batch_states([obs], agent.device, agent.phi))
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break