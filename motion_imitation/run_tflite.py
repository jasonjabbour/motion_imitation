import os
import inspect
from re import I

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from unicodedata import name
import gym
import tensorflow as tf
# import tflite_runtime.interpreter as tflite
import numpy as np
import pickle
import argparse
from mpi4py import MPI
import csv
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.animation
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib import transforms

import freeze_graph
from motion_imitation.envs import env_builder as env_builder
from motion_imitation.robots import bittle
from motion_imitation import bittle_insticts

import pathlib

try: 
    from motion_imitation.serialMaster.policy2serial import *
except Exception as e:
    print('Serial Master Import Error. Make sure Bittle is connected to a COM Port', e)

ENABLE_ENV_RANDOMIZER = False
INITIAL_POSE = np.array([.52,.52,.52,.52,.52,.52,.52,.52])

saved_actions_lsts = []
saved_motor_joint_angles_lsts = []
saved_smoothed_actions_lsts = []

gaits_ax = None

imu_data_dict = {'imu_x':[],'imu_y':[],'imu_z':[],'imu_dx':[],'imu_dy':[],'imu_dz':[],
                 'imu_index':[], 
                 #'rotated_imu_x':[],'rotated_imu_y':[],'rotated_imu_z':[],'imu_Wx':[],'imu_Wy':[],'imu_Wz':[]
                 }
joint_coordinates = {}

DEAD_ZONE = True
DEAD_ZONE_AMOUNT = .0175 #1deg
LARGEST_SHOULDER_ALLOWED_AMOUNT = 30 #deg
SMALLEST_KNEE_ALLOWED_AMOUNT = 30 #deg
LARGEST_KNEE_ALLOWED_AMOUNT = 50 #deg
LAST_ACTION = None
SHRINK_ERROR = np.array([])

def test_tflite(env, model_dir, model_number, verify=False, angle_analysis=False, imu_data=False, see_gaits=False, real_bittle=False, quantized=False):
    '''Use tflite to make predictions and compare predictions to stablebaselines output'''
    global imu_data_dict, saved_actions_lsts

    #TFlite model Path
    tflite_model = model_dir + '/bittle_frozen_model'+ args.model_number + ".tflite"
    if quantized:
       tflite_model = model_dir + '/bittle_frozen_model'+ args.model_number + "_quantized.tflite" 

    #Load tflite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model, experimental_delegates=None)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    steps = 400
    done = False

    if verify:
        #Get verification saved obs and actions
        with open(model_dir + '/saved_info_1ep_model' + model_number + '.pickle','rb') as handle:
            saved_info = pickle.load(handle)
        #Length of saved info
        steps = len(saved_info['obs'])
    elif angle_analysis:
        obs = env.reset()
        _robot = env.robot

        delta_lst = []
        previous_saved_action = [0]*8 
        steps = 1000
    elif real_bittle:
        initializeCommands()
        #initialize pose
        step_real_bittle(INITIAL_POSE, convert=True)
        time.sleep(2)

        last_action_queue = np.concatenate((INITIAL_POSE, [0]*16))
        imu_sensor_real = getBittleIMUSensorInfo()
        obs = np.concatenate((imu_sensor_real,last_action_queue))

        current_imu_data = obs[0:18]

        steps = 1000
        obs = env.reset()
        _robot = env.robot
    else:
        obs = env.reset()
        _robot = env.robot
        current_imu_data = obs[0:18]
        time.sleep(15)

    for step in range(steps):

        if imu_data:
    
                imu_data_dict['imu_x'].append(current_imu_data[0])
                imu_data_dict['imu_y'].append(current_imu_data[1])
                imu_data_dict['imu_z'].append(current_imu_data[2])
                imu_data_dict['imu_dx'].append(current_imu_data[3])
                imu_data_dict['imu_dy'].append(current_imu_data[4])
                imu_data_dict['imu_dz'].append(current_imu_data[5])
                imu_data_dict['imu_index'].append(step)

        if verify: 
            #Get observation from saved file
            obs = saved_info['obs'][step]
    
        #Use TFlite model to make predictions
        input_data = obs.reshape(1, -1)
        input_data = np.array(input_data, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'],input_data)

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.array(output_data).flatten()
        action = output_data[:8]

        #Show output of TFLite model vs SB Saved Output
        if verify: 
            calculate_error(action,saved_info['actions'][step])
            print("TFLite Output:",action)
            print("SB Saved Output",saved_info['actions'][step])
            print('\n')
        #Generate observations based on Pybullet Simulation
        else:
            if (not real_bittle):
                #action -> processed action:
                #trajectory wrapper adds initial pose to predicted raw action: action+[0.52 0.52 0.52 0.52 0.52 0.52 0.52 0.52]
                obs, reward, done, info = env.step(action)
                #Get Processed Action
                processed_action = obs[18:26] #equal to man_prc_action
                current_imu_data = obs[0:18]

                #Get actual joint angle position of legs  
                saved_motor_joint_angles_lsts.append(_robot.GetMotorAngles())

            #Process Action
            processed_action = manually_process_action(action)
            #Save Processed Action
            saved_actions_lsts.append(processed_action)  

            #Smooth the processed actions
            smoothed_action = smooth_action(processed_action)
            #Save the smoothed actions
            saved_smoothed_actions_lsts.append(smoothed_action)

            if real_bittle:
                step_real_bittle(smoothed_action, convert=True)
                time.sleep(.1)

                imu_sensor_real = getBittleIMUSensorInfo()
                #print('Real IMU Data :',imu_sensor_real[0:6])

                last_action_queue = np.concatenate((smoothed_action, last_action_queue))
                last_action_queue = last_action_queue[0:24]

                obs = np.concatenate((imu_sensor_real,last_action_queue))

                current_imu_data = imu_sensor_real
            elif done:
                #obs = env.reset()
                pass
                            
            if angle_analysis:
                for joint, angle in enumerate(processed_action):
                    delta = abs(np.degrees(angle)-previous_saved_action[joint])
                    if delta >= np.degrees(DEAD_ZONE_AMOUNT) :
                        delta_lst.append(delta)
                    previous_saved_action = action
    
    if verify:
        total_error = 0
        for error in SHRINK_ERROR:
            total_error+=error
        
        print(SHRINK_ERROR.size)
        print('Error between SB Model and TFLite Model is:',total_error/SHRINK_ERROR.size)
        
    if imu_data:
        plot_imu_data()
        save_imu_data_csv()

    if see_gaits:
        plot_gaits(plot_sim=True, plot_true_sim_motor_angles=True, plot_smoothed_actions=True)
        plot_gaits_timeseries(plot_sim=True)

    #Plot histogram of joint angle change 
    if angle_analysis:
        plot_anlge_analysis(delta_lst)

def manually_process_action(raw_action):
    '''Process action manually'''
    global LAST_ACTION

    #trajectory wrapper
    manually_proc_action = np.array(raw_action) + INITIAL_POSE

    #dead zone  where motor is not moved if action is less than 1 deg difference
    if DEAD_ZONE == True and LAST_ACTION is not None:
        for joint_num, angle in enumerate(manually_proc_action):
            if abs(angle-LAST_ACTION[joint_num]) <= DEAD_ZONE_AMOUNT: # or abs(angle-LAST_ACTION[joint_num]) >= LARGEST_ALLOWED_AMOUNT : #1deg
                manually_proc_action[joint_num] = LAST_ACTION[joint_num]

    LAST_ACTION = manually_proc_action
    return manually_proc_action

def smooth_action(action):
    '''Smooth action prediction using a trailing moving average'''

    smoothed_action = action
    window_size = 4

    #Start computing moving average once enough data is stored
    for i in range(len(saved_actions_lsts) - window_size):
        #Store previous joint angles in windows
        window = np.array(saved_actions_lsts[i:i+window_size])
        #Calculate average of current window for each joint (sum over rows)
        smoothed_action = window.sum(axis=0)/window_size

    return smoothed_action

def plot_anlge_analysis(delta_lst):
    '''Plot the change of joint angle

    delta_lst: A list of joint angle changes'''

    plt.hist(delta_lst, bins=30)
    plt.show()
    print(f'The average angle change is: {sum(delta_lst)/len(delta_lst)}')


def plot_gaits(plot_sim=False, plot_real=False, plot_true_sim_motor_angles=False, plot_smoothed_actions=False):
    '''Runner function for plotting the gaits the right legs
        Uses the saved action data from test_tflite() function
        
        plot_sim: plot the gaits of the pybullet bittle using the predicted processed actions
        plot_real: plot the gaits of the real life bittle
        plot_true_sim_motor_angles: plot the actual motor angles from the simulation
        
        plot_gaits() -> update_gaits() -> get_transform_gaits()'''

    global gaits_ax, joint_coordinates

    #Initialize Figure
    fig, gaits_ax = plt.subplots()
    plt.xlim([0,250])
    plt.ylim([0,250])
    
    #Set leg width and height
    width, height = 3, 40

    #Right Front Joints
    joint9_x, joint9_y = (150, 100)
    joint13_x, joint13_y = (joint9_x, joint9_y-height)

    #Right Back Joints
    joint10_x, joint10_y = (50, 100)
    joint14_x, joint14_y = (joint10_x, joint10_y-height)

    #Torso Constant
    gaits_ax.add_patch(Rectangle((joint10_x,joint10_y), joint9_x-joint10_x , 3, fill=True,
        facecolor='peachpuff', edgecolor='darkorange', lw=1, angle=0))
    
    #Initialize list for collecting end effector coordinates for leg
    joint_coordinates['Right_Front_Leg_Predicted_Processed'] = []

    #Plot the true poisitions of the legs
    if plot_true_sim_motor_angles:
        #Right Front Top
        gaits_ax.add_patch(Rectangle((joint9_x,joint9_y), width, -height,
            fill=True, facecolor='lightgrey', edgecolor='silver', lw=1, angle=0))

        #Right Front Bottom
        gaits_ax.add_patch(Rectangle((joint13_x,joint13_y), height, width,
            fill=True, facecolor='lightgrey', edgecolor='silver', lw=1, angle=0))
        
        #Right Back Top
        gaits_ax.add_patch(Rectangle((joint10_x,joint10_y), width, -height, fill=True,
            facecolor='lightgrey', edgecolor='silver', lw=1, angle=0))

        #Right Back Bottom
        gaits_ax.add_patch(Rectangle((joint14_x,joint14_y), height, width, fill=True, 
            facecolor='lightgrey', edgecolor='silver', lw=1, angle=0))
        
        #Initialize list for collecting end effector coordinates for leg
        joint_coordinates['Right_Front_Leg_True_Motor_Angle'] = []

    #Plot the smoothed actions
    if plot_smoothed_actions:
        #Right Front Top
        gaits_ax.add_patch(Rectangle((joint9_x,joint9_y), width, -height,
            fill=True, facecolor='lightcoral', edgecolor='red', lw=1, angle=0))

        #Right Front Bottom
        gaits_ax.add_patch(Rectangle((joint13_x,joint13_y), height, width,
            fill=True, facecolor='lightcoral', edgecolor='red', lw=1, angle=0))
        
        #Right Back Top
        gaits_ax.add_patch(Rectangle((joint10_x,joint10_y), width, -height, fill=True,
            facecolor='lightcoral', edgecolor='red', lw=1, angle=0))

        #Right Back Bottom
        gaits_ax.add_patch(Rectangle((joint14_x,joint14_y), height, width, fill=True, 
            facecolor='lightcoral', edgecolor='red', lw=1, angle=0))

        #Initialize list for collecting end effector coordinates for leg
        joint_coordinates['Right_Front_Leg_Smoothed_Actions'] = []

    #Plot Initial Leg Positions
    #Right Front Top
    gaits_ax.add_patch(Rectangle((joint9_x,joint9_y), width, -height, fill=True,
        facecolor='peachpuff', edgecolor='darkorange', lw=1, angle=0))

    #Right Front Bottom
    gaits_ax.add_patch(Rectangle((joint13_x,joint13_y), height, width, fill=True,
        facecolor='peachpuff', edgecolor='darkorange', lw=1, angle=0))
    
    #Right Back Top
    gaits_ax.add_patch(Rectangle((joint10_x,joint10_y), width, -height, fill=True,
        facecolor='peachpuff', edgecolor='darkorange', lw=1, angle=0))

    #Right Back Bottom
    gaits_ax.add_patch(Rectangle((joint14_x,joint14_y), height, width, fill=True,
        facecolor='peachpuff', edgecolor='darkorange', lw=1, angle=0))
    
    ani = matplotlib.animation.FuncAnimation(fig, update_gaits, frames=len(saved_actions_lsts)-1, fargs=(plot_true_sim_motor_angles,plot_smoothed_actions), repeat=False, interval=1, cache_frame_data=False)

    #Add Legend
    legend_patches = []
    legend_patches.append(mpatches.Patch(color='peachpuff', label='Predicted Processed Actions'))
    if plot_true_sim_motor_angles:
        legend_patches.append(mpatches.Patch(color='lightgrey', label='True Motor Angles'))
    if plot_smoothed_actions:
        legend_patches.append(mpatches.Patch(color='lightcoral', label='Smoothed Actions'))

    plt.legend(handles=legend_patches)
    plt.show()

count = 0
def update_gaits(step, plot_true_sim_motor_angles, plot_smoothed_actions):
    '''Helper function for Plot Gaits'''
    global gaits_ax, joint_coordinates,count

    if count == 1:
        time.sleep(0)
    count+=1

    #clear old joint patches (one for each joint except for torso)
    for patch in range(len(gaits_ax.patches)-1):
        gaits_ax.patches.pop()

    #Plot the predicted processed joint angles and store x,y coordinates of leg end effectors
    sim_x_coords_lst, sim_y_coords_lst = get_transform_gaits(saved_actions_lsts[step], 'peachpuff', 'darkorange')
    #Save the coordinates (x,y) of the end effector of the front leg
    joint_coordinates['Right_Front_Leg_Predicted_Processed'].append((sim_x_coords_lst[0],sim_y_coords_lst[0]))
    
    #Plot the Tips of the Bottom Legs for the predicted processed joint angles
    gaits_ax.scatter(sim_x_coords_lst, sim_y_coords_lst, color='blue', alpha=.3)

    if plot_true_sim_motor_angles:
        #Plot the true sim motor angles and store x,y coordinates of leg end effectors
        true_motor_angles_x_coords_lst, true_motor_angles_y_coords_lst = get_transform_gaits(saved_motor_joint_angles_lsts[step], 'lightgrey', 'silver')
        #Save the coordinates (x,y) of the end effector of the front leg
        joint_coordinates['Right_Front_Leg_True_Motor_Angle'].append((true_motor_angles_x_coords_lst[0],true_motor_angles_y_coords_lst[0]))
    
        #Plot the Tips of the Bottom Legs for the true sim motor angles
        gaits_ax.scatter(true_motor_angles_x_coords_lst, true_motor_angles_y_coords_lst, color='grey', alpha=.2)

    if plot_smoothed_actions:
        #Plot the true sim motor angles and store x,y coordinates of leg end effectors
        smoothed_actions_x_coords_lst, smoothed_actions_y_coords_lst = get_transform_gaits(saved_smoothed_actions_lsts[step], 'lightcoral', 'red')
        #Save the coordinates (x,y) of the end effector of the front leg
        joint_coordinates['Right_Front_Leg_Smoothed_Actions'].append((smoothed_actions_x_coords_lst[0],smoothed_actions_y_coords_lst[0]))
    
        #Plot the Tips of the Bottom Legs for the true sim motor angles
        gaits_ax.scatter(smoothed_actions_x_coords_lst, smoothed_actions_y_coords_lst, color='red', alpha=.2)
            
            
def get_transform_gaits(action, facecolor, edgecolor):
    '''Moves the rectangle representation of bittle using the a list of actions that represent joint angles. 
        Calculates the end effector x,y position of the bottom of each leg. 

        Current implementation of the function always plots a total full two legs.
            -> Right Front Leg
            -> Right Back Leg 

        return: a list of x coordinates and a list of y coordinates for the end effector position of each leg
    '''
    width, height = 3, 40
    x_front, y_front = (150, 100)
    x_back, y_back = (50, 100)

    # --- Right Front Leg ---
    #Right Front Top (Joint 9)
    joint_9_angle = -action[0] #opposite to pybullet direction, keep in radians
    joint_9_rect = Rectangle((x_front,y_front), width, -height,
                    fill=True,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    lw=1)
    joint_9_transform = transforms.Affine2D().rotate_around(x_front,y_front,joint_9_angle) #rad
    joint_9_rect.set_transform(joint_9_transform+gaits_ax.transData)
    gaits_ax.add_patch(joint_9_rect)
    joint_9_coords = joint_9_rect.get_patch_transform().transform(joint_9_rect.get_path().vertices[:-1])
    joint_9_x_bottom = joint_9_transform.transform(joint_9_coords)[3][0]
    joint_9_y_bottom = joint_9_transform.transform(joint_9_coords)[3][1]

    #Right Front Bottom (Joint 13)
    joint_13_angle = -action[1] + joint_9_angle #must add action 0 to pretend that the joint is impacted by parent joint
    joint_13_rect = Rectangle((joint_9_x_bottom,joint_9_y_bottom), height, width,
                    fill=True,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    lw=1)
    joint_13_transform = transforms.Affine2D().rotate_around(joint_9_x_bottom,joint_9_y_bottom,joint_13_angle) #rad
    joint_13_rect.set_transform(joint_13_transform+gaits_ax.transData)
    gaits_ax.add_patch(joint_13_rect)
    joint_13_coords = joint_13_rect.get_patch_transform().transform(joint_13_rect.get_path().vertices[:-1])
    joint_13_x_bottom = joint_13_transform.transform(joint_13_coords)[1][0]
    joint_13_y_bottom = joint_13_transform.transform(joint_13_coords)[1][1]

    # --- Right Back Leg ---
    #Right Back Top (Joint 10)
    joint_10_angle = -action[4] #opposite to pybullet direction, keep in radians
    joint_10_rect = Rectangle((x_back,y_back), width, -height,
                    fill=True,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    lw=1)
    joint_10_transform = transforms.Affine2D().rotate_around(x_back,y_back,joint_10_angle) #rad
    joint_10_rect.set_transform(joint_10_transform+gaits_ax.transData)
    gaits_ax.add_patch(joint_10_rect)
    joint_10_coords = joint_10_rect.get_patch_transform().transform(joint_10_rect.get_path().vertices[:-1])
    joint_10_x_bottom = joint_10_transform.transform(joint_10_coords)[3][0]
    joint_10_y_bottom = joint_10_transform.transform(joint_10_coords)[3][1]

    #Right Back Bottom (Joint 14)
    joint_14_angle = -action[5] + joint_10_angle #must add action 0 to pretend that the joint is impacted by parent joint
    joint_14_rect = Rectangle((joint_10_x_bottom,joint_10_y_bottom), height, width,
                    fill=True,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    lw=1)
    joint_14_transform = transforms.Affine2D().rotate_around(joint_10_x_bottom,joint_10_y_bottom,joint_14_angle) #rad
    joint_14_rect.set_transform(joint_14_transform+gaits_ax.transData)
    gaits_ax.add_patch(joint_14_rect)
    joint_14_coords = joint_14_rect.get_patch_transform().transform(joint_14_rect.get_path().vertices[:-1])
    joint_14_x_bottom = joint_14_transform.transform(joint_14_coords)[1][0]
    joint_14_y_bottom = joint_14_transform.transform(joint_14_coords)[1][1]

    #x and y for the Front Leg and Back Leg
    return [joint_13_x_bottom, joint_14_x_bottom], [joint_13_y_bottom,joint_14_y_bottom]

def plot_gaits_timeseries(plot_sim=False):
    '''Plot the predicted actions y coordinate over time for the end effector coordinates of a leg'''

    colors = ['darkorange','silver','red']

    #Initialize Figure according to the number of saved joint coordinate keys in dictionary
    fig, gait_timeseries_ax = plt.subplots(len(joint_coordinates))
    
    #Plot the timeseries for each leg in joint_coordinates dictionary
    for i, leg_name_type in enumerate(joint_coordinates):
        
        #Joint Coordinate lists hold data as [(x,y),(x,y),...]
        steps_lst = list(range(len(joint_coordinates[leg_name_type])))
        x_coords = []
        y_coords = []

        #Save (x,y) coordinates in to independent lists
        for step in steps_lst:
            x_coords.append(joint_coordinates[leg_name_type][step][0])
            y_coords.append(joint_coordinates[leg_name_type][step][1])

        #Plot the data for this leg over time
        gait_timeseries_ax[i].plot(steps_lst, y_coords, color=colors[i%len(colors)]) 
        gait_timeseries_ax[i].set_xlim([0,len(steps_lst)])
        gait_timeseries_ax[i].set_ylim([0,150])
        gait_timeseries_ax[i].set_xlabel('Step')
        gait_timeseries_ax[i].set_ylabel('Y Coordinate of Leg')
        gait_timeseries_ax[i].set_title('Leg Gaits over time for ' + leg_name_type)


    #Add spacing between subplots
    plt.subplots_adjust(hspace=.6)
    plt.show()     

def plot_imu_data():
    '''Plot Scatter Plots of the IMU Data for Simulated Bittle or Real Bittle
    
    Will use whatever data stored in imu_data_dict'''

    fig, ax = plt.subplots(6, figsize=(10,6))
    
    ax[0].scatter(x=imu_data_dict['imu_index'],y=imu_data_dict['imu_x'])
    ax[0].set_xlabel("Step")
    ax[0].set_ylabel("Roll")
    ax[0].set_ylim([-1.5,1.5])

    ax[1].scatter(x=imu_data_dict['imu_index'],y=imu_data_dict['imu_y'])
    ax[1].set_xlabel("Step")
    ax[1].set_ylabel("Pitch")
    ax[1].set_ylim([-1.5,1.5])

    ax[2].scatter(x=imu_data_dict['imu_index'],y=imu_data_dict['imu_z'])
    ax[2].set_xlabel("Step")
    ax[2].set_ylabel("Yaw")
    ax[2].set_ylim([-1.5,1.5])

    ax[3].scatter(x=imu_data_dict['imu_index'],y=imu_data_dict['imu_dx'])
    ax[3].set_xlabel("Step")
    ax[3].set_ylabel("Roll Rate")
    ax[3].set_ylim([-1.5,1.5])

    ax[4].scatter(x=imu_data_dict['imu_index'],y=imu_data_dict['imu_dy'])
    ax[4].set_xlabel("Step")
    ax[4].set_ylabel("Pitch Rate")
    ax[4].set_ylim([-4,4])

    ax[5].scatter(x=imu_data_dict['imu_index'],y=imu_data_dict['imu_dz'])
    ax[5].set_xlabel("Step")
    ax[5].set_ylabel("Yaw Rate")
    ax[5].set_ylim([-1.5,1.5])

    plt.show()

def save_imu_data_csv():

    df = pd.DataFrame(imu_data_dict)
    df.to_csv('imu_data_file_temp.csv')
    print("Data Saved to CSV File!")


def calculate_error(action, saved_action):
    global SHRINK_ERROR 
    error = abs(action - saved_action)
    SHRINK_ERROR = np.concatenate((SHRINK_ERROR, error))


def quantize_tflite(model_dir, model_number, frozen_model_output_layer):
    '''Float16 Quantization'''

    frozen_model_name = model_dir + '/bittle_frozen_model'+model_number
    path = frozen_model_name+".pb"

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(path, input_arrays=['input/Ob'], output_arrays=[frozen_model_output_layer])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    tflite_save_file = frozen_model_name+"_quantized.tflite"
    with open(tflite_save_file, 'wb') as f:
            f.write(tflite_model)

    print(f"\n>>Model Converted to Quantized TFLite using Float16 Quantization: {tflite_save_file} <<\n")


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


def step_real_bittle(action, reorder=True, clip_action=True, convert=False):
    # change actions to degrees
    if convert:
        action = np.degrees(action)
    
    if clip_action:
        #SHOULDER
        if action[2] >= LARGEST_SHOULDER_ALLOWED_AMOUNT:
            print(action[2])
            action[2] = LARGEST_SHOULDER_ALLOWED_AMOUNT
        elif action[2] <= -LARGEST_SHOULDER_ALLOWED_AMOUNT:
            print(action[2])
            action[2] = -LARGEST_SHOULDER_ALLOWED_AMOUNT
        if action[0] >= LARGEST_SHOULDER_ALLOWED_AMOUNT:
            print(action[0])
            action[0] = LARGEST_SHOULDER_ALLOWED_AMOUNT
        elif action[0] <= -LARGEST_SHOULDER_ALLOWED_AMOUNT:
            print(action[0])
            action[0] = -LARGEST_SHOULDER_ALLOWED_AMOUNT
        
        #KNEE
        if action[1] < SMALLEST_KNEE_ALLOWED_AMOUNT:
            action[1] = SMALLEST_KNEE_ALLOWED_AMOUNT
        elif action[1] > LARGEST_KNEE_ALLOWED_AMOUNT:
            action[1] = LARGEST_KNEE_ALLOWED_AMOUNT
        if action[3] < SMALLEST_KNEE_ALLOWED_AMOUNT:
            action[3] = SMALLEST_KNEE_ALLOWED_AMOUNT
        elif action[3] > LARGEST_KNEE_ALLOWED_AMOUNT:
            action[3] = LARGEST_KNEE_ALLOWED_AMOUNT

    if reorder:
        # set all joint angles simultaneously
        task = ['i',[9,action[0],13,action[1],8,action[2],12,action[3],10, action[4],14,action[5],11, action[6],15, action[7]],0]
    else:
        task = ['i',[8,action[0],9,action[1],10,action[2],11,action[3],12, action[4],13,action[5],14, action[6],15, action[7]],0]
    sendCommand(task)


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
    arg_parser.add_argument("--angle_analysis",dest="angle_analysis", action="store_true", default=False)
    arg_parser.add_argument("--imu_data",dest="imu_data", action="store_true", default=False)
    arg_parser.add_argument("--see_gaits",dest="see_gaits", action="store_true", default=False)
    arg_parser.add_argument("--real_bittle",dest="real_bittle", action="store_true", default=False)
    arg_parser.add_argument("--quantized_model",dest="quantized_model", action="store_true", default=False)

    args = arg_parser.parse_args()

    num_procs = MPI.COMM_WORLD.Get_size()
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

    model_dir = 'output/all_model'+args.model_number

    if args.mode == 'full_pipeline':
        #Freeze the TF Model
        tf_2_frozen(model_dir, args.model_number, args.frozen_model_output_layer)
        #Convert the frozen TF model to tflite
        frozen_pb_2_tflite(model_dir, args.model_number, args.frozen_model_output_layer)
    
    elif args.mode == 'quantize':
        quantize_tflite(model_dir, args.model_number, args.frozen_model_output_layer)        

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
        test_tflite(env, model_dir, args.model_number, args.verify, args.angle_analysis, 
                    args.imu_data, args.see_gaits, args.real_bittle, args.quantized_model)
    
    elif args.mode == 'calibrate':

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

        #calibrate(env)
    



