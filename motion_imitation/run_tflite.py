import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import sys
from unicodedata import name
import gym
import tensorflow as tf
# import tflite_runtime.interpreter as tflite
import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse

from mpi4py import MPI
from motion_imitation.envs import env_builder as env_builder
from motion_imitation.robots import bittle

ENABLE_ENV_RANDOMIZER = False

def test_tflite(env, tflite_model='bittle_frozen_axis1'):
    '''Use tflite to make predictions and compare predictions to stablebaselines output'''

    #TFlite model path
    model_save_file = tflite_model + ".tflite"

    #Load tflite model
    interpreter = tf.lite.Interpreter(model_path=model_save_file, experimental_delegates=None)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #Get verification saved obs and actions
    with open('bittle_controller/data/saved_info_1ep.pickle','rb') as handle:
        saved_info = pickle.load(handle)
    
    difference = []
    index = []

    obs = env.reset()
    

    for i in range(len(saved_info['obs'])):
        # obs = saved_info['obs'][i]

        # #REMOVE: Replace last 60 obs with saved 60
        # obs = np.concatenate((obs[:60], saved_info['obs'][i][:60])) 

        input_data = obs.reshape(1, -1)
        input_data = np.array(input_data, dtype=np.float32)
        # print("Observation",input_data)
        interpreter.set_tensor(input_details[0]['index'],input_data)

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.array(output_data).flatten()
        print("TFLite Output:",output_data)
        # print("SB Saved Output",saved_info['actions'][i])
        print('\n')

        # for g in range(len(output_data)):
        #     difference.append(output_data[g] - saved_info['actions'][i][0][g] )
        #     index.append(i)

        obs, reward, done, info = env.step(output_data[:8])

        # if done:
        #     obs = env.reset()

    plt.scatter(index,difference)
    plt.show()


def frozen_pb_2_tflite(frozen_pb_path, frozen_model_output_layer):
    '''Convert a frozen tensorflow protobuf model to tflite'''
    path = frozen_pb_path+".pb"

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(path, input_arrays=['input/Ob'], output_arrays=[frozen_model_output_layer])
    tflite_model = converter.convert()

    tflite_save_file = frozen_pb_path+".tflite"
    with open(tflite_save_file, 'wb') as f:
            f.write(tflite_model)

    print(f"Model Converted to TFLite:{tflite_save_file}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--mode", dest="mode", type=str, default='full_pipeline')
    arg_parser.add_argument("--frozen_model_file", dest="frozen_model_file", type=str, default='output/bittle_frozen_axis1')
    arg_parser.add_argument("--frozen_model_output_layer", dest="frozen_model_output_layer", type=str, default='chicken')
    arg_parser.add_argument("--tflite_model", dest="tflite_model", type=str, default='output/bittle_frozen_axis1')
    arg_parser.add_argument("--env", dest="env", type=str, default='bittle')
    arg_parser.add_argument("--motion_file", dest="motion_file", type=str, default='motion_imitation/data/motions/pace_bittle.txt')
    arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=True)

    args = arg_parser.parse_args()

    num_procs = MPI.COMM_WORLD.Get_size()
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

    if args.env == 'bittle':
        robot_class = bittle.Bittle
        robot = bittle

    #Create env and pass in bittle as robot and robot class
    enable_env_rand = ENABLE_ENV_RANDOMIZER 
    env = env_builder.build_imitation_env(motion_files=[args.motion_file],
                                        num_parallel_envs=num_procs,
                                        mode='test',
                                        enable_randomizer=enable_env_rand,
                                        enable_rendering=args.visualize,
                                        robot_class=robot_class, 
                                        robot=robot)

    if args.mode == 'frozen2tflite':
        frozen_pb_2_tflite(args.frozen_model_file, args.frozen_model_output_layer)
    if args.mode == 'test_tflite':
        test_tflite(env, args.tflite_model)



