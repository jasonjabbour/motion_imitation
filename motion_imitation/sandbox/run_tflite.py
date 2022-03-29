import sys
import gym
import tflite_runtime.interpreter as tflite
#from bittle_env import BittleEnv
import numpy as np
import pickle
import matplotlib.pyplot as plt


if __name__ == '__main__':
    env_name = 'MountainCarContinuous-v0'
    model_prefix = 'bittle_frozen_axis1'
    model_save_file = model_prefix + ".tflite"

    # env = gym.make(env_name)
    #env = BittleEnv(True)
    #obs = env.reset()

    interpreter = tflite.Interpreter(model_path=model_save_file, experimental_delegates=None)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    with open('saved_info_1ep.pickle','rb') as handle:
        saved_info = pickle.load(handle)
    
    difference = []
    index = []

    for i in range(len(saved_info['obs'])):
        obs = saved_info['obs'][i]

        input_data = obs.reshape(1, -1)
        input_data = np.array(input_data, dtype=np.float32)
        # print("Observation",input_data)
        interpreter.set_tensor(input_details[0]['index'],input_data)

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.array(output_data).flatten()
        print("TFLite Output:",output_data)
        print("SB Saved Output",saved_info['actions'][i])
        print('\n')

        # for g in range(len(output_data)):
        #     difference.append(output_data[g] - saved_info['actions'][i][0][g] )
        #     index.append(i)

        #obs, reward, done, info = env.step(output_data)

        # env.render()
        # if done:
        #     obs = env.reset()

    plt.scatter(index,difference)
    plt.show()

import tensorflow as tf

path = "bittle_frozen_axis1.pb"

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(path, input_arrays=['input/Ob'], output_arrays=['chicken'])
tflite_model = converter.convert()

tflite_save_file = "bittle_frozen_axis1.tflite"
with open(tflite_save_file, 'wb') as f:
        f.write(tflite_model)

print("Model Converted to TFLite")