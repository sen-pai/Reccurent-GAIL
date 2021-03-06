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
# from pfrl.wrappers import atari_wrappers


logging.basicConfig(level=20)


env_name = "MiniGrid-Empty-5x5-v0"

cenv = gym.make(env_name)
cenv = wrappers.FlatObsWrapper(cenv)


obs_size = cenv.reset().shape[0]
n_actions = cenv.action_space.n

def make_env(idx, test):
    # Use different random seeds for train and test envs
    process_seed = 1
    env_seed = 1
    env = gym.make(env_name)
    env = wrappers.FlatObsWrapper(env)
    print('env made')
    env.seed(env_seed)
    if True:
        env = pfrl.wrappers.Monitor(
            env, 'mlp_run', mode="evaluation" if test else "training"
        )
    if True:
        env = pfrl.wrappers.Render(env)
    return env

def make_batch_env(test):
    vec_env = pfrl.envs.MultiprocessVectorEnv(
        [
            (lambda: make_env(idx, test))
            for idx, env in enumerate(range(1))
        ]
    )

    return vec_env



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
)

if __name__ == '__main__':
    pfrl.experiments.train_agent_batch(
        agent,
        env= make_batch_env(False),
        steps = 50000,
        outdir = 'mlp_run',
        checkpoint_freq=None,
        log_interval=10,
        max_episode_len=None,
        step_offset=0,
        evaluator=None,
        successful_score=None,
        step_hooks=(),
        return_window_size=10,
        logger=None,
    )
