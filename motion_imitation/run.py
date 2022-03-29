# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

import os
import inspect
from statistics import mode
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import argparse
from mpi4py import MPI
import numpy as np
import os
import random
import tensorflow as tf
import time

import math
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from motion_imitation.envs import env_builder as env_builder
from motion_imitation.learning import imitation_policies as imitation_policies
from motion_imitation.learning import ppo_imitation as ppo_imitation
from motion_imitation.robots import bittle
from motion_imitation.robots import laikago

from stable_baselines.common.callbacks import CheckpointCallback

from serialMaster.policy2serial import *
from sandbox.stable_2_pytorch import *

from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile


TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 256

ENABLE_ENV_RANDOMIZER = True

#For stats
PREVIOUS_ACTION = []
CHANGE_BANK = []
SAVED_ACTIONS = []
SAVED_ACTIONS_LISTS = []
SAVED_OBS = []
FILTER_ACTION_BANK = {0:{'paused':None, 'previous_action':None,'previous_direction':None},
  1:{'paused':None, 'previous_action':None,'previous_direction':None},
  2:{'paused':None, 'previous_action':None,'previous_direction':None},
  3:{'paused':None, 'previous_action':None,'previous_direction':None},
  4:{'paused':None, 'previous_action':None,'previous_direction':None},
  5:{'paused':None, 'previous_action':None,'previous_direction':None},
  6:{'paused':None, 'previous_action':None,'previous_direction':None},
  7:{'paused':None, 'previous_action':None,'previous_direction':None}}

SAVED_ACTIONS_PYTORCH = []

def set_rand_seed(seed=None):
  if seed is None:
    seed = int(time.time())

  seed += 97 * MPI.COMM_WORLD.Get_rank()

  tf.set_random_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  return

def build_model(env, num_procs, timesteps_per_actorbatch, optim_batchsize, output_dir):
  # MLP Policy with two layers of size 512 and 256
  policy_kwargs = {
      "net_arch": [{"pi": [512, 256],
                    "vf": [512, 256]}],
      "act_fun": tf.nn.relu
  }

  timesteps_per_actorbatch = int(np.ceil(float(timesteps_per_actorbatch) / num_procs))
  optim_batchsize = int(np.ceil(float(optim_batchsize) / num_procs))

  model = ppo_imitation.PPOImitation(
               policy=imitation_policies.ImitationPolicy,
               env=env,
               gamma=0.95,
               timesteps_per_actorbatch=timesteps_per_actorbatch,
               clip_param=0.2,
               optim_epochs=1,
               optim_stepsize=1e-5,
               optim_batchsize=optim_batchsize,
               lam=0.95,
               adam_epsilon=1e-5,
               schedule='constant',
               policy_kwargs=policy_kwargs,
               tensorboard_log=output_dir,
               verbose=1)
  return model


def train(model, env, total_timesteps, output_dir="", int_save_freq=0):
  if (output_dir == ""):
    save_path = None
  else:
    save_path = os.path.join(output_dir, "model4_cutobs.zip")
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
  

  callbacks = []
  # Save a checkpoint every n steps
  if (output_dir != ""):
    if (int_save_freq > 0):
      int_dir = os.path.join(output_dir, "model4_cutobs_intermedate")
      callbacks.append(CheckpointCallback(save_freq=int_save_freq, save_path=int_dir,
                                          name_prefix='model4_cutobs'))

  model.learn(total_timesteps=total_timesteps, save_path=save_path, callback=callbacks)

  return

def test(model, env, num_procs, num_episodes=None):
  curr_return = 0
  sum_return = 0
  episode_count = 0

  if num_episodes is not None:
    num_local_episodes = int(np.ceil(float(num_episodes) / num_procs))
  else:
    num_local_episodes = np.inf

  o = env.reset()
  # with model.graph.as_default():
  while episode_count < num_local_episodes:
    #save observation
    save_obs(o)

    a, _ = model.predict(o, deterministic=True)
    o, r, done, info = env.step(a)
    curr_return += r

    #Save action
    save_action(a)
    #Save action as a list of list
    save_action_lists(a)
    #Get delta of actions
    calculate_angle_change(a)


    if done:
        o = env.reset()
        sum_return += curr_return
        episode_count += 1

  sum_return = MPI.COMM_WORLD.allreduce(sum_return, MPI.SUM)
  episode_count = MPI.COMM_WORLD.allreduce(episode_count, MPI.SUM)

  mean_return = sum_return / episode_count

  if MPI.COMM_WORLD.Get_rank() == 0:
      print("Mean Return: " + str(mean_return))
      print("Episode Count: " + str(episode_count))
  
  # print(CHANGE_BANK)
  # print(SAVED_ACTIONS)
  # print(SAVED_OBS)
  # graph_angle_change(CHANGE_BANK)
  # graph_direction_change(CHANGE_BANK)

  #CHECK IF OBS of last 60 are the same: graph four target poses
  # last_60 = []
  # for obs_lst in SAVED_OBS:
  #   obs_lst = list(obs_lst)
  #   last_60+=obs_lst[60:]

  # print(sum(last_60))
  # index_lst = list(range(len(last_60)))
  # plt.scatter(index_lst[:len(last_60)//num_local_episodes], last_60[:len(last_60)//num_local_episodes], c='blue')
  # plt.scatter(index_lst[len(last_60)//num_local_episodes:], last_60[len(last_60)//num_local_episodes:], c='red')

  # #Add labels
  # plt.xlabel("i")
  # plt.ylabel("Pose")
  # #Show plot
  # plt.show()

  #SAVE INFO
  # saved_info_dict={'obs':SAVED_OBS,'actions':SAVED_ACTIONS_LISTS}
  # print(SAVED_ACTIONS_LISTS)

  # with open('saved_info_1000steps.pickle', 'wb') as handle:
  #   pickle.dump(saved_info_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return

def save_action(action):
  '''Save a list of all actions made'''
  global SAVED_ACTIONS
  for a in action:
    SAVED_ACTIONS.append(a)
  
def save_action_lists(action_lst):
  '''Save a list of actions made as a list of lists'''
  global SAVED_ACTIONS_LISTS
  SAVED_ACTIONS_LISTS.append(action_lst)

def save_obs(obs):
  '''Save a list of all observations'''
  global SAVED_OBS
  SAVED_OBS.append(obs)

def calculate_angle_change(action):
  '''Calculate the change between actions'''
  global PREVIOUS_ACTION, CHANGE_BANK

  if len(PREVIOUS_ACTION) >= 8:
    for i in range(len(PREVIOUS_ACTION)):
      # calculate the change in each joint angle and add to bank
      delta = PREVIOUS_ACTION[i] - action[i]
      # convert to degrees
      delta = math.degrees(delta)
      CHANGE_BANK.append(delta)
    
  PREVIOUS_ACTION = action


def graph_angle_change(delta_lst):
  ''' Take list of angle deltas and plots a histogram'''

  with plt.style.context('seaborn-pastel'):
    #Create Histogram
    fig, axs = plt.subplots(figsize = (6,4), alpha=.65)
    axs.hist(delta_lst, bins=40)

    #Style
    plt.style.use("bmh")

    #Labels
    plt.xlabel("Joint Angle Change (Degrees)")
    plt.ylabel("Count")
    plt.title("Distribution of Joint Angle Changes")

    #Show plot
    plt.show()

def graph_direction_change(delta_lst):
  '''Graph the change direction of angle'''
      
  #list with 8 lists for each joint
  each_joint_delta = []
  

  #break up into separate joints
  for i in range(0,8):
    each_joint_delta.append(delta_lst[i::8])
  
  #create index list
  index_lst = list(range(0,len(each_joint_delta[0])))

  joint_key =  ['9','13','8','12','10','14','11','15']
  joint = 1

  pointer = each_joint_delta[joint][0]
  color = 'red'
  arrow_style = 'v'

  #loop through the joint angles of one leg
  for i in range(len(index_lst)-1):
    #Save pointer
    old_pointer = pointer
    #make vertical line
    x_graph = [index_lst[i], index_lst[i]]
    #start at last position to new position
    y_graph = [pointer, pointer + each_joint_delta[joint][i+1]]
    pointer+= each_joint_delta[joint][i+1]

    # based on direction choose color of line
    if old_pointer > pointer:
        color = 'firebrick'
        arrow_style = '-v'
    else:
        color = 'mediumseagreen'
        arrow_style = '-^'
    
    plt.plot(x_graph, y_graph,arrow_style, color=color)

  #Add labels
  plt.xlabel("t")
  plt.ylabel("Joint Position (Degrees)")
  plt.title("Direction of Joint Angle (Joint " + joint_key[joint] + ")")
  #Show plot
  plt.show()
  
    
def test_sim2real(model, env, num_procs, num_episodes=None):
  ''' Test policy on the real-life bittle '''
  curr_return = 0
  sum_return = 0
  episode_count = 0

  if num_episodes is not None:
    num_local_episodes = int(np.ceil(float(num_episodes) / num_procs))
  else:
    num_local_episodes = np.inf

  initializeCommands()
  #time.sleep(5)
  o = env.reset()
  while episode_count < num_local_episodes:
    a, _ = model.predict(o, deterministic=True)

    #step the simulation
    o, r, done, info = env.step(a)
    #get the processed action calculated during simulation
    proc_action = info['processed_action']

    #call to change motor angles of bittle
    step_real_bittle(ser, proc_action)

    #rest
    time.sleep(.007)

    #get imu data
    real_joint_angles, imu_sensor = getBittleIMUSensorInfo()

    #IMU: 0-11, Last Action: 12-35, Motor Angle: 36-59, Target: 60 - 119
    o = np.concatenate((imu_sensor, o[12:36], real_joint_angles, o[60:]))


    curr_return += r

    if done:
        o = env.reset()
        sum_return += curr_return
        episode_count += 1

  sum_return = MPI.COMM_WORLD.allreduce(sum_return, MPI.SUM)
  episode_count = MPI.COMM_WORLD.allreduce(episode_count, MPI.SUM)

  mean_return = sum_return / episode_count

  if MPI.COMM_WORLD.Get_rank() == 0:
      print("Mean Return: " + str(mean_return))
      print("Episode Count: " + str(episode_count))

  return

def step_real_bittle(ser, action):
  ''' Send the action as serial commands to real-life bittle
      joints order key: '9','13','8','12','10','14','11','15'
  '''
  # change actions to degrees
  action = np.degrees(action)

  filter = False

  if filter:
    make_change, action = filter_actions(action)

    key = [9,13,8,12,10,14,11,15]
    send_actions = []
    
    #construct actions
    for i in range(len(make_change)):
      if make_change[i]:
        send_actions.append(key[i])
        send_actions.append(action[i])

    if len(send_actions) > 0:
      # set all joint angles simultaneously
      task = ['i',send_actions,0]
      sendCommand(task)
  else:
    # set all joint angles simultaneously
    task = ['i',[9,action[0],13,action[1],8,action[2],12, action[3], 10, action[4],14, action[5],11, action[6],15, action[7]],0]
    sendCommand(task)

def filter_actions(action):
  '''Filter actions that change directions or are small changes'''
  global FILTER_ACTION_BANK

#  1:{'paused':None, 'previous_action':None,'previous_direction':None},
  make_change = [False]*8
  for i in range(len(action)):
    action_symbol = direction2symbol(action[i])
    # if paused too long or first step, then make change
    if (FILTER_ACTION_BANK[i]['previous_action'] == None) or (FILTER_ACTION_BANK[i]['paused'] > 5):
      make_change[i] = True
      FILTER_ACTION_BANK[i]['paused'] = 0
      FILTER_ACTION_BANK[i]['previous_action'] = action[i]
      FILTER_ACTION_BANK[i]['previous_direction'] = action_symbol
    # if current change is smaller than 10 degrees then don't make any change
    elif abs(FILTER_ACTION_BANK[i]['previous_action'] - action[i]) < 5:
      # increase paused
      FILTER_ACTION_BANK[i]['paused'] = FILTER_ACTION_BANK[i]['paused'] + 1
    else:
      make_change[i] = True
          
  return make_change, action

def direction2symbol(angle):
  if angle < 0:
    return '-'
  return '+'

def calibrate(env):
  pass

def graph_model_differences():
  """Plot the difference of prediction values for two models"""
  diff = []
  index = []
  for i in range(len(SAVED_ACTIONS_PYTORCH)-8):
    diff.append(math.degrees(abs(SAVED_ACTIONS[i+8]) - abs(SAVED_ACTIONS_PYTORCH[i+8])))
    index.append(i)

  plt.plot(index, diff,'o')

  #Add labels
  plt.xlabel("t")
  plt.ylabel("Difference (degrees)")
  plt.title("Difference in Predicted Actions between Stable Baselines Policy and Pytorch Policy")
  #Show plot
  plt.show()
      

def main():
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--seed", dest="seed", type=int, default=None)
  arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
  arg_parser.add_argument("--motion_file", dest="motion_file", type=str, default="motion_imitation/data/motions/dog_pace.txt")
  arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
  arg_parser.add_argument("--output_dir", dest="output_dir", type=str, default="output")
  arg_parser.add_argument("--num_test_episodes", dest="num_test_episodes", type=int, default=None)
  arg_parser.add_argument("--model_file", dest="model_file", type=str, default="")
  arg_parser.add_argument("--total_timesteps", dest="total_timesteps", type=int, default=2e8)
  arg_parser.add_argument("--int_save_freq", dest="int_save_freq", type=int, default=0) # save intermediate model every n policy steps
  arg_parser.add_argument("--robot", dest="specified_robot", type=str, default="laikago")

  # If True, returns a 60 dim obs space instead of 120 dim. The 4 future reference frames are cut (15 each)
  # Change is made to imitation_wrapper_env.py
  arg_parser.add_argument("--obs_cut_future_frames", dest="obs_cut_future_frames", action="store_true", default=False)

  args = arg_parser.parse_args()
  
  num_procs = MPI.COMM_WORLD.Get_size()
  os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

  if args.specified_robot == "bittle":
    robot_class = bittle.Bittle
    robot = bittle
  else:
    robot_class = laikago.Laikago
    robot = laikago

  #Create env and pass in bittle as robot and robot class
  enable_env_rand = ENABLE_ENV_RANDOMIZER and (args.mode != "test")
  env = env_builder.build_imitation_env(motion_files=[args.motion_file],
                                        num_parallel_envs=num_procs,
                                        mode=args.mode,
                                        enable_randomizer=enable_env_rand,
                                        enable_rendering=args.visualize,
                                        robot_class=robot_class, 
                                        robot=robot,
                                        obs_cut_future_frames=args.obs_cut_future_frames)
  model = build_model(env=env,
                      num_procs=num_procs,
                      timesteps_per_actorbatch=TIMESTEPS_PER_ACTORBATCH,
                      optim_batchsize=OPTIM_BATCHSIZE,
                      output_dir=args.output_dir)

  
  if args.model_file != "":
    model.load_parameters(args.model_file)
 
  if args.mode == "train":
    train(model=model, 
            env=env, 
            total_timesteps=args.total_timesteps,
            output_dir=args.output_dir,
            int_save_freq=args.int_save_freq)
  elif args.mode == "test":
    test(model=model,
           env=env,
           num_procs=num_procs,
           num_episodes=args.num_test_episodes)
  elif args.mode == "sim2real":
    # run policy on the real life bittle
    test_sim2real(model=model,
      env=env,
      num_procs=num_procs,
      num_episodes=args.num_test_episodes) 
  elif args.mode == "convert":    
    #Access the model's parameters
    # for key, value in model.get_parameters().items():
    #       print(key, value.shape)
    # model/pi_fc0/w:0 (120, 512)
    # model/pi_fc0/b:0 (512,)
    # model/vf_fc0/w:0 (120, 512)
    # model/vf_fc0/b:0 (512,)
    # model/pi_fc1/w:0 (512, 256)
    # model/pi_fc1/b:0 (256,)
    # model/vf_fc1/w:0 (512, 256)
    # model/vf_fc1/b:0 (256,)
    # model/vf/w:0 (256, 1)
    # model/vf/b:0 (1,)
    # model/pi/w:0 (256, 8)
    # model/pi/b:0 (8,)
    # model/q/w:0 (256, 8)
    # model/q/b:0 (8,)

    #Stable Baselines Model Action Probabilities
    # obs = env.reset()
    # #does not make sense for continuous actions
    # print(model.action_probability(obs)) #https://github.com/hill-a/stable-baselines/issues/126

    #Convert stable baselines policy to tensorflow:
    with model.graph.as_default():
      # print(model.get_parameter_list())
      # print(model.get_parameters())
      print(model.policy_pi.policy_proba[0])
      print(model.policy_pi.policy_proba[1])
      tf.saved_model.simple_save(model.sess, 'model2_tf_test8_axis1', inputs={"obs":model.policy_pi.obs_ph},
        outputs={"action": tf.concat(model.policy_pi.policy_proba, axis=1, name="chicken")})

    print('Model successfully converted.')

  elif args.mode == "calibrate":
    calibrate(env)
  else:
      assert False, "Unsupported mode: " + args.mode

  return

if __name__ == '__main__':
  main()
