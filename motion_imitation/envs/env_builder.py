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
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from motion_imitation.envs import locomotion_gym_env
from motion_imitation.envs import locomotion_gym_config
from motion_imitation.envs.env_wrappers import imitation_wrapper_env
from motion_imitation.envs.env_wrappers import observation_dictionary_to_array_wrapper
from motion_imitation.envs.env_wrappers import observation_dictionary_to_array_wrapper as obs_dict_to_array_wrapper
from motion_imitation.envs.env_wrappers import trajectory_generator_wrapper_env
from motion_imitation.envs.env_wrappers import simple_openloop
from motion_imitation.envs.env_wrappers import simple_forward_task
from motion_imitation.envs.env_wrappers import imitation_task
from motion_imitation.envs.env_wrappers import default_task

from motion_imitation.envs.sensors import environment_sensors
from motion_imitation.envs.sensors import sensor_wrappers
from motion_imitation.envs.sensors import robot_sensors
from motion_imitation.envs.utilities import controllable_env_randomizer_from_config
from motion_imitation.robots import laikago
from motion_imitation.robots import a1
from motion_imitation.robots import bittle
from motion_imitation.robots import robot_config



def build_laikago_env( motor_control_mode, enable_rendering):

  sim_params = locomotion_gym_config.SimulationParameters()
  sim_params.enable_rendering = enable_rendering
  sim_params.motor_control_mode = motor_control_mode
  sim_params.reset_time = 2
  sim_params.num_action_repeat = 10
  sim_params.enable_action_interpolation = False
  sim_params.enable_action_filter = False
  sim_params.enable_clip_motor_commands = False
  
  gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params)

  robot_class = laikago.Laikago

  sensors = [
      robot_sensors.MotorAngleSensor(num_motors=laikago.NUM_MOTORS),
      robot_sensors.IMUSensor(),
      environment_sensors.LastActionSensor(num_actions=laikago.NUM_MOTORS)
  ]

  task = default_task.DefaultTask()

  env = locomotion_gym_env.LocomotionGymEnv(gym_config=gym_config, robot_class=robot_class,
                                            robot_sensors=sensors, task=task)

  #env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
  #env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(env,
  #                                                                     trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(action_limit=laikago.UPPER_BOUND))

  return env


def build_imitation_env(motion_files, num_parallel_envs, mode,
                        enable_randomizer, enable_rendering,
                        robot_class=laikago.Laikago,
                        robot=laikago,
                        trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(action_limit=laikago.UPPER_BOUND),
                        obs_cut_future_frames=False):

  assert len(motion_files) > 0

  if robot_class == bittle.Bittle:
    #get_action() returns initial pose of .52 + action
    trajectory_generator=simple_openloop.BittlePoseOffsetGenerator(action_limit=bittle.UPPER_BOUND)

  curriculum_episode_length_start = 20
  # curriculum_episode_length_end = 600
  # curriculum_episode_length_end = 300
  curriculum_episode_length_end = 500

  #Set Simulation Parameters
  sim_params = locomotion_gym_config.SimulationParameters()
  sim_params.enable_rendering = enable_rendering
  sim_params.allow_knee_contact = True
  sim_params.motor_control_mode = robot_config.MotorControlMode.POSITION
  #make changes parameters to simulation here if you would like:
  #sim_params.sim_time_step_s = .07

  #Group the simulation parameters into the gym_config class
  gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params)

  #Use the sensors of the robot
  sensors = [
      # reads motor angles from robot minutaur class: GetMotorAngles(). Applies STD for noise. History of last 3 angles thus 3*8 = 24
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.MotorAngleSensor(num_motors=robot.NUM_MOTORS), num_history=3),
      # [Roll, Pitch, Roll Rate, Pitch Rate] * 3 = 12
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=robot_sensors.IMUSensor(), num_history=3),
      # Position of the motors with a bitterworth filter applied. [8]*3 = 24
      sensor_wrappers.HistoricSensorWrapper(wrapped_sensor=environment_sensors.LastActionSensor(num_actions=robot.NUM_MOTORS), num_history=3)
  ]

  #Calculates reward base on the difference of pose (angle of each joint), velocity (velocity of each joint), end effector (base position and base rotation), root pose, and root velocity of sim bittle and reference motion bittle
  task = imitation_task.ImitationTask(ref_motion_filenames=motion_files,
                                      enable_cycle_sync=True,
                                      tar_frame_steps=[1, 2, 10, 30],
                                      ref_state_init_prob=0.9,
                                      warmup_time=0.25)

  #Domain Randomization
  randomizers = []
  if enable_randomizer:
    randomizer = controllable_env_randomizer_from_config.ControllableEnvRandomizerFromConfig(verbose=False)
    randomizers.append(randomizer)

  #Initialize Open AI Gym Environment
  env = locomotion_gym_env.LocomotionGymEnv(gym_config=gym_config, robot_class=robot_class,
                                            env_randomizers=randomizers, robot_sensors=sensors, task=task)

  #Flattens observations of individual sensors into 1 array of length 60
  env = observation_dictionary_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
  #Flattens the target observations
  env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(env,
                                                                       trajectory_generator=trajectory_generator)

  if mode == "test":
      curriculum_episode_length_start = curriculum_episode_length_end

  #Modifies observational space. Adds the 60 target observations
  env = imitation_wrapper_env.ImitationWrapperEnv(env,
                                                  episode_length_start=curriculum_episode_length_start,
                                                  episode_length_end=curriculum_episode_length_end,
                                                  curriculum_steps=30000000,
                                                  num_parallel_envs=num_parallel_envs,
                                                  cut_future_frames=obs_cut_future_frames)
  return env



def build_regular_env(robot_class,
                      motor_control_mode,
                      enable_rendering=False,
                      on_rack=False,
                      action_limit=(0.75, 0.75, 0.75),
                      wrap_trajectory_generator=True):

  sim_params = locomotion_gym_config.SimulationParameters()
  sim_params.enable_rendering = enable_rendering
  sim_params.motor_control_mode = motor_control_mode
  sim_params.reset_time = 2
  sim_params.num_action_repeat = 10
  sim_params.enable_action_interpolation = False
  sim_params.enable_action_filter = False
  sim_params.enable_clip_motor_commands = False
  sim_params.robot_on_rack = on_rack

  gym_config = locomotion_gym_config.LocomotionGymConfig(
      simulation_parameters=sim_params)

  sensors = [
      robot_sensors.BaseDisplacementSensor(),
      robot_sensors.IMUSensor(),
      robot_sensors.MotorAngleSensor(num_motors=a1.NUM_MOTORS),
  ]

  task = simple_forward_task.SimpleForwardTask()

  env = locomotion_gym_env.LocomotionGymEnv(gym_config=gym_config,
                                            robot_class=robot_class,
                                            robot_sensors=sensors,
                                            task=task)

  env = obs_dict_to_array_wrapper.ObservationDictionaryToArrayWrapper(
      env)
  if (motor_control_mode
      == robot_config.MotorControlMode.POSITION) and wrap_trajectory_generator:
    if robot_class == laikago.Laikago:
      env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(
          env,
          trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(
              action_limit=action_limit))
    elif robot_class == a1.A1:
      env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(
          env,
          trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(
              action_limit=action_limit))
  return env