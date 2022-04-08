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

import freeze_graph

from mpi4py import MPI
from motion_imitation.envs import env_builder as env_builder
from motion_imitation.robots import bittle

ENABLE_ENV_RANDOMIZER = False

def test_tflite(env, model_dir, model_number, verify=False):
    '''Use tflite to make predictions and compare predictions to stablebaselines output'''

    #TFlite model Path
    tflite_model = model_dir + '/bittle_frozen_model'+ args.model_number + ".tflite"

    #Load tflite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model, experimental_delegates=None)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if verify:
        #Get verification saved obs and actions
        with open(model_dir + '/saved_info_1ep_model' + model_number + '.pickle','rb') as handle:
            saved_info = pickle.load(handle)
        #Length of saved info
        steps = len(saved_info['obs'])
    else:
        obs = env.reset()
        steps = 500
    
    for i in range(steps):

        if verify: 
            #Get observation from saved file
            obs = saved_info['obs'][i]
    
        #Use TFlite model to make predictions
        input_data = obs.reshape(1, -1)
        input_data = np.array(input_data, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'],input_data)

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.array(output_data).flatten()

        #Show output of TFLite model vs SB Saved Output
        if verify: 
            print("TFLite Output:",output_data)
            print("SB Saved Output",saved_info['actions'][i])
            print('\n')

        #Generate observations based on Pybullet Simulation
        else:
            obs, reward, done, info = env.step(output_data[:8])

            if done:
                obs = env.reset()


def frozen_pb_2_tflite(model_dir, model_number, frozen_model_output_layer):
    '''Convert a frozen tensorflow protobuf model to tflite'''

    frozen_model_name = model_dir + '/bittle_frozen_model'+model_number
    path = frozen_model_name+".pb"

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(path, input_arrays=['input/Ob'], output_arrays=[frozen_model_output_layer])
    tflite_model = converter.convert()

    tflite_save_file = frozen_model_name+".tflite"
    with open(tflite_save_file, 'wb') as f:
            f.write(tflite_model)

    print(f"\n>>Model Converted to TFLite: {tflite_save_file} <<\n")

def tf_2_frozen(model_dir, model_number, output_layer):
    '''Freeze a tensorflow protobuf model '''

    input_graph = model_dir + '/model'+model_number+'_tf/saved_model.pb'
    input_saved_model_dir = model_dir + '/model'+model_number+'_tf'
    input_binary = True
    output_graph =  model_dir + '/bittle_frozen_model' + model_number + '.pb'
    output_node_names = output_layer
    input_saver = input_checkpoint = initializer_nodes = variable_names_whitelist = variable_names_blacklist = ''
    restore_op_name = 'save/restore_all'
    filename_tensor_name = 'save/Const:0'
    clear_devices = True
    freeze_graph.freeze_graph(input_graph,input_saver, input_binary,
                            input_checkpoint, output_node_names,
                            restore_op_name, filename_tensor_name, 
                            output_graph, clear_devices, initializer_nodes,
                            variable_names_whitelist, variable_names_blacklist, 
                            input_saved_model_dir=input_saved_model_dir)
    
    print(f'\n>> TF Protobuf was successfully frozen: {output_graph} <<\n')

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--mode", dest="mode", type=str, default='full_pipeline')
    arg_parser.add_argument("--frozen_model_output_layer", dest="frozen_model_output_layer", type=str, default='chicken')
    arg_parser.add_argument("--env", dest="env", type=str, default='bittle')
    arg_parser.add_argument("--motion_file", dest="motion_file", type=str, default='motion_imitation/data/motions/pace_bittle.txt')
    arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=True)
    arg_parser.add_argument("--obs_cut_future_frames", dest="obs_cut_future_frames", action="store_true", default=False)
    arg_parser.add_argument("--model_number",dest="model_number", type=str, default='7')
    arg_parser.add_argument("--verify",dest="verify", action="store_true", default=False)
    args = arg_parser.parse_args()

    num_procs = MPI.COMM_WORLD.Get_size()
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

    model_dir = 'output/all_model'+args.model_number

    if args.mode == 'full_pipeline':
        #Freeze the TF Model
        tf_2_frozen(model_dir, args.model_number, args.frozen_model_output_layer)
        #Convert the frozen TF model to tflite
        frozen_pb_2_tflite(model_dir, args.model_number, args.frozen_model_output_layer)

    elif args.mode == 'test_tflite': 

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
                                            robot=robot,
                                            obs_cut_future_frames=args.obs_cut_future_frames)
        #run TFlite model
        test_tflite(env, model_dir, args.model_number, args.verify)



