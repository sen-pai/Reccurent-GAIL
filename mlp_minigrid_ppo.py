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

cenv = gym.make("MiniGrid-Empty-6x6-v0")
cenv = wrappers.FlatObsWrapper(cenv)


obs_size = cenv.reset().shape[0]
n_actions = cenv.action_space.n

def make_env(idx, test):
    # Use different random seeds for train and test envs
    process_seed = 1
    env_seed = 1
    env = gym.make("MiniGrid-LavaCrossingS9N1-v0")
    env = wrappers.FlatObsWrapper(env)
    print('env made')
    env.seed(env_seed)
    if True:
        env = pfrl.wrappers.Monitor(
            env, 'mlp_run', mode="evaluation" if test else "training"
        )
    if False:
        env = pfrl.wrappers.Render(env)
    return env

def make_batch_env(test):
    vec_env = pfrl.envs.MultiprocessVectorEnv(
        [
            (lambda: make_env(idx, test))
            for idx, env in enumerate(range(2))
        ]
    )

    return vec_env



model = nn.Sequential(
    nn.Linear(obs_size, 512),
    nn.ReLU(),
    pfrl.nn.Branched(
        # action branch
        nn.Sequential(nn.Linear(512, n_actions), SoftmaxCategoricalHead(),),
        # value branch
        nn.Linear(512, 1),
    ),
)


# model = pfrl.nn.RecurrentSequential(
#         lecun_init(nn.Conv2d(obs_n_channels, 32, 8, stride=4)),
#         nn.ReLU(),
#         lecun_init(nn.Conv2d(32, 64, 4, stride=2)),
#         nn.ReLU(),
#         lecun_init(nn.Conv2d(64, 64, 3, stride=1)),
#         nn.ReLU(),
#         nn.Flatten(),
#         lecun_init(nn.Linear(3136, 512)),
#         nn.ReLU(),
#         lecun_init(nn.GRU(num_layers=1, input_size=512, hidden_size=512)),
#         pfrl.nn.Branched(
#             nn.Sequential(
#                 lecun_init(nn.Linear(512, n_actions), 1e-2),
#                 SoftmaxCategoricalHead(),
#             ),
#             lecun_init(nn.Linear(512, 1)),
#         ),
#     )


opt = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-5)


# mostly not needed
def phi(x):
    # Feature extractor
    return np.asarray(x, dtype=np.float32)


agent = PPO(
    model,
    opt,
    gpu=0,
    phi=phi,
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
        steps = 10000,
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
