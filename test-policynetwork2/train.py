#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#-*- coding: utf-8 -*-

import os
import gym
import numpy as np
import parl
from time import sleep

from agent import Agent
from model import Model
from algorithm import PolicyGradient  # from parl.algorithms import PolicyGradient

from parl.utils import logger

LEARNING_RATE = 0.01

np.random.seed(2)  # reproducible
N_STATES = 6   # the length of the 1 dimensional world
ACTIONS = ['left', 'right']     # available actions
FRESH_TIME = 0.1    # fresh time for one move

def get_env_feedback(State, A):
    # This is how agent will interact with the environment
    done=False
    S=State[0]
    #print("State=", State, "S=", S, "A=", A)
    if A == 0:    # move right
        if S == N_STATES - 2:   # terminate
            S_ = S + 1
            R = 1
            done=True
        else:
            S_ = S + 1
            R = 0
    else:   # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    #print("S_=", S_, "R=", R)
    next_obs=(S_, 0)
    return next_obs, R, done


def update_env(State, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    S=State[0]
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        sleep(1)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        sleep(FRESH_TIME)

# 训练一个episode
def run_episode(agent, episode):
    obs_list, action_list, reward_list = [], [], []
    step_counter = 0
    obs = (0,0)
    update_env(obs, episode, step_counter)
    while True:
        obs_list.append(obs)
        action = agent.sample(obs)
        action_list.append(action)

        obs, reward, done = get_env_feedback(obs, action)
        #print("next_obs=", obs, "action=", action, "reward=", reward)
        reward_list.append(reward)
        update_env(obs, episode, step_counter)
        step_counter += 1
        if done:
            break
    print("total_steps=",step_counter)
    return obs_list, action_list, reward_list


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(agent):
    eval_reward = []
    for i in range(5):
        step_counter = 0
        obs = (0, 0)
        update_env(obs, i, step_counter)
        episode_reward = 0
        while True:
            action = agent.predict(obs)
            obs, reward, done = get_env_feedback(obs,action)
            update_env(obs, i, step_counter)
            step_counter += 1
            #episode_reward += reward
            if done:
                break
        print("   i=",i,"step_counter=",step_counter)
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def calc_reward_to_go(reward_list, gamma=1.0):
    for i in range(len(reward_list) - 2, -1, -1):
        # G_i = r_i + γ·G_i+1
        reward_list[i] += gamma * reward_list[i + 1]  # Gt
    return np.array(reward_list)


def main():
    action_dim = 2
    observation_dim = 2
    logger.info('observation_dim {}, action_dim {}'.format(observation_dim, action_dim))

    model = Model(act_dim=action_dim)
    alg = PolicyGradient(model, lr=LEARNING_RATE)
    agent = Agent(alg, obs_dim=observation_dim, act_dim=action_dim)

    # 加载模型
    #if os.path.exists('./model.ckpt'):
    #    agent.restore('./model.ckpt')
    #    run_episode(env, agent, train_or_test='test', render=True)
    #    exit()

    for i in range(10):
        obs_list, action_list, reward_list = run_episode(agent, i)
        if i % 2 == 0:
            logger.info("Episode {}, Reward Sum {}.".format(
                i, sum(reward_list)))

        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)
        batch_reward = calc_reward_to_go(reward_list)

        agent.learn(batch_obs, batch_action, batch_reward)
        if (i + 1) % 10 == 0:
            total_reward = evaluate(agent)
            logger.info('Test reward: {}'.format(total_reward))

    # save the parameters to ./model.ckpt
    agent.save('./model.ckpt')


if __name__ == '__main__':
    main()
