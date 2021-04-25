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


# import matplotlib.pyplot as plt
# from imitation.data.types import Trajectory

# from imitation.util import logger, util
# from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
# from gym.wrappers.frame_stack import FrameStack



env_name = "MiniGrid-Empty-5x5-v0"

env = gym.make(env_name)
env = wrappers.FlatObsWrapper(env)


obs_size = env.reset().shape[0]
n_actions = env.action_space.n

# def make_env(idx, test):
#     # Use different random seeds for train and test envs
#     env_seed = idx
#     env = gym.make(env_name)
#     env = wrappers.FlatObsWrapper(env)
#     print('env made')
#     env.seed(env_seed)
#     if True:
#         env = pfrl.wrappers.Monitor(
#             env, 'rnn_run', mode="evaluation" if test else "training"
#         )
#     if True:
#         env = pfrl.wrappers.Render(env)
#     return env

# def make_batch_env(test):
#     vec_env = pfrl.envs.MultiprocessVectorEnv(
#         [
#             (lambda: make_env(idx, test))
#             for idx, env in enumerate(range(8))
#         ]
#     )

#     return vec_env



model = pfrl.nn.RecurrentSequential(
    nn.Linear(obs_size, 64),
    nn.ReLU(),
    nn.GRU(num_layers=1, input_size=64, hidden_size=64),
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
    # phi=phi,
    update_interval=2048,
    minibatch_size=64,
    epochs=10,
    clip_eps=0.1,
    clip_eps_vf=None,
    standardize_advantages=True,
    entropy_coef=1e-2,
    recurrent=True,
    max_grad_norm=0.5,
    # max_recurrent_sequence_len = 50,
)


agent.load("rnn_run/50000_finish/")

print("agent loaded")
# if __name__ == '__main__':
#     pfrl.experiments.train_agent_batch(
#         agent,
#         env= make_batch_env(False),
#         steps = 50000,
#         outdir = 'rnn_run',
#         checkpoint_freq=None,
#         log_interval=10,
#         max_episode_len=None,
#         step_offset=0,
#         evaluator=None,
#         successful_score=None,
#         step_hooks=(),
#         return_window_size=10,
#         logger=None,
#     )


# env = wrappers.FlatObsWrapper(gym.make('MiniGrid-Empty-Random-6x6-v0'))
# env = gym.make('MiniGrid-Empty-Random-6x6-v0')
# env = wrappers.RGBImgObsWrapper(env)
# env = wrappers.ImgObsWrapper(env)


# venv = util.make_vec_env('MiniGrid-Empty-Random-6x6-v0', n_envs=1, post_wrappers= [wrappers.RGBImgObsWrapper, wrappers.ImgObsWrapper])
# env = util.make_vec_env('MiniGrid-Empty-Random-6x6-v0', n_envs=1, post_wrappers= [wrappers.FlatObsWrapper, FrameStack], post_wrappers_kwargs=[{}, {"num_stack":10}])

# env = VecTransposeImage(venv)
# env = VecTransposeImage(DummyVecEnv(env))

# model = PPO.load("ppo_stack4_minigrid_empty")

# print(env.reset().shape)
traj_dataset = []
for traj in range(30):
    # obs_list = []
    # action_list = []
    # info_list = []
    obs = env.reset()
    for i in range(500):
        action, _state = agent.act([obs])
        obs, reward, done, info = env.step(action)
        # action_list.append(action[0])
        # info_list.append({})
        # obs_list.append(obs.reshape(-1))
        env.render()
        if done:
            break
    # traj_dataset.append(Trajectory(obs = np.array(obs_list), acts= np.array(action_list), infos = np.array(info_list)))


# with open('empty_6_stack4.pkl', 'wb') as handle:
#     pickle.dump(traj_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
