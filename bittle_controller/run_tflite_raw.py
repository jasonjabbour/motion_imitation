
import tflite_runtime.interpreter as tflite

import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import argparse

try: 
    from serialMaster.policy2serial import *
except Exception as e:
    print('Serial Master Import Error. Make sure Bittle is connected to a COM Port', e)


INITIAL_POSE = np.array([.52,.52,.52,.52,.52,.52,.52,.52])
saved_actions_lsts = []
saved_motor_joint_angles_lsts = []
saved_smoothed_actions_lsts = []

DEAD_ZONE = True
DEAD_ZONE_AMOUNT = .0175 #1deg
SMALLEST_SHOULDER_ALLOWED_AMOUNT = -90 #deg
LARGEST_SHOULDER_ALLOWED_AMOUNT = 90 #deg
SMALLEST_KNEE_ALLOWED_AMOUNT = 0 #deg
LARGEST_KNEE_ALLOWED_AMOUNT = 90 #deg
LAST_ACTION = None

def verify_tflite(interpreter, saved_info, show_output=False):
    '''Use tflite to make predictions and compare predictions to stablebaselines output
    
    interpreter: Loaded frozen tflite model
    saved_info: saved obs and actions from SB model which will be used to compare to tflite output
    show_output: plot the results
    '''
    #Prepare tflite model
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i in range(len(saved_info['obs'])):
        #Read Observations
        obs = saved_info['obs'][i]

        input_data = obs.reshape(1, -1)
        input_data = np.array(input_data, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'],input_data)

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.array(output_data).flatten()

        #Display output of TFLite and SB Models
        if show_output:
            print("TFLite Output:", output_data)
            print("SB Saved Output", saved_info['actions'][i])
            print('\n')


def model_processing_time(tflite_model_dir, plot_time=False, trials=5):
    '''#Run 5 Different Saved Info Files 5 Times
        Each info file consists of two episodes with 500 steps each'''
    
    trial_process_times_lst = []

    #Run 5 Different Saved Info Files 5 Times
    for i in range(trials):
        for i in range(trials):
            #Get saved info to pass into model
            file_number = i + 1
            #UPDATE FILE NAME TO BE MODEL SPECIFIC
            saved_info = get_saved_info('data/saved_info_2ep_1000steps' + str(file_number))

            #Load the model
            tflite_interpreter = load_tflite_model(tflite_model_dir)

            #Start Process timer
            start_process_time = time.process_time()
            #Run model for the saved observations
            verify_tflite(tflite_interpreter,saved_info)
            #Calculate Process Time
            trial_process_time = time.process_time()-start_process_time
            
            #Save Times
            trial_process_times_lst.append(trial_process_time)
    
    #Calculate Average Time
    average_time = 0
    for i in range(len(trial_process_times_lst)):
        average_time+=trial_process_times_lst[i]
    average_time/=(trials*trials)
    print('Average Time:',average_time)

    #Plot
    if plot_time:
        plt.scatter(list(range(len(trial_process_times_lst))) ,trial_process_times_lst)
        plt.title('TFLite Time to process 1000 Steps for 25 Trials')
        plt.xlabel('Trial')
        plt.ylabel('Time (s)')
        plt.savefig('captures/tflite_time_test.png')


def deploy_on_bittle(interpreter):
    '''Deploy model on real life bittle and receive feedback
    
    interpreter: Loaded frozen tflite model
    '''
    global saved_actions_lsts

    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #initialize connection
    initializeCommands()
    #initialize pose
    step_real_bittle(INITIAL_POSE, convert=True)
    time.sleep(2)

    last_action_queue = np.concatenate((INITIAL_POSE, [0]*16))
    imu_sensor_real = getBittleIMUSensorInfo()
    obs = np.concatenate((imu_sensor_real,last_action_queue))

    steps = 1000

    for step in range(steps):
    
        #Use TFlite model to make predictions
        input_data = obs.reshape(1, -1)
        input_data = np.array(input_data, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'],input_data)

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.array(output_data).flatten()
        action = output_data[:8]

        #Process Action
        processed_action = manually_process_action(action)
        #Save Processed Action
        saved_actions_lsts.append(processed_action) 

        #Smooth the processed actions
        smoothed_action = smooth_action(processed_action)
        #Save the smoothed actions
        saved_smoothed_actions_lsts.append(smoothed_action)

        #Send action to real bittle
        step_real_bittle(smoothed_action, convert=True)

        #Sleep
        time.sleep(.1)

        #Read IMU Data
        imu_sensor_real = getBittleIMUSensorInfo()
        print(imu_sensor_real)

        #Create last action queue
        last_action_queue = np.concatenate((smoothed_action, last_action_queue))
        last_action_queue = last_action_queue[0:24]

        #Create observations from imu and last actions
        obs = np.concatenate((imu_sensor_real,last_action_queue))  

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
    
def step_real_bittle(action, reorder=True, clip_action=True, convert=False):
    # change actions to degrees
    if convert:
        action = np.degrees(action)
    
    if clip_action:
        #SHOULDER
        if action[2] >= LARGEST_SHOULDER_ALLOWED_AMOUNT:
            action[2] = LARGEST_SHOULDER_ALLOWED_AMOUNT
        elif action[2] <= SMALLEST_SHOULDER_ALLOWED_AMOUNT:
            action[2] = SMALLEST_SHOULDER_ALLOWED_AMOUNT
        if action[0] >= LARGEST_SHOULDER_ALLOWED_AMOUNT:
            action[0] = LARGEST_SHOULDER_ALLOWED_AMOUNT
        elif action[0] <= SMALLEST_SHOULDER_ALLOWED_AMOUNT:
            action[0] = SMALLEST_SHOULDER_ALLOWED_AMOUNT
        
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

def manually_control_bittle():
    '''Manually send commands to bittle'''
 
    #initialize commands
    initializeCommands()

    while True:
        action = [52]*8
        index = int(input('Which index would you like to change? (0-7)'))
        if index >= 0 and index < 8:
            joint_angle = float(input('What angle would you like to move joint to?'))
            action[index] = joint_angle
            task = ['i',[9,action[0],13,action[1],8,action[2],12, action[3], 10, action[4],14, action[5],11, action[6],15, action[7]],0]
            sendCommand(task)
            print("Command sent")
            time.sleep(1)
        else:
            print('Invalid index')

def load_tflite_model(model_save_file):
    '''Load and return a TFLite model
    
    model_save_file: TFlite model path
    '''
    #Load tflite model
    interpreter = tflite.Interpreter(model_path=model_save_file, experimental_delegates=None)

    return interpreter    

def get_saved_info(file_name):
    #Get verification saved obs and actions
    with open(file_name+'.pickle','rb') as handle:
        saved_info = pickle.load(handle)
    return saved_info

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--mode", dest="mode", type=str, default='test')
    arg_parser.add_argument("--model_number", dest="model_number", type=str, default='13')
    arg_parser.add_argument("--verification_info", dest="verification_info", type=str, default='data/saved_info_1ep')
    args = arg_parser.parse_args()

    model_dir = 'output/all_model'+args.model_number
    tflite_model_dir = model_dir + '/bittle_frozen_model' + args.model_number + '.tflite'
    mode = args.mode

    if mode == 'time':
        #Plot the timing results 
        plot_time_answer = False
        plot_time_answer = input("Would you like to plot the time for each trial? [Y/N] ")
        if plot_time_answer.lower() == "yes" or plot_time_answer.lower() == "y":
            plot_time_answer = True

        #Time the processing time of the tflite model
        model_processing_time(tflite_model_dir, plot_time_answer)

    elif mode == 'verify':
        #Get saved verification data
        saved_info = get_saved_info(args.verification_info)

        #Load the model
        tflite_interpreter = load_tflite_model(tflite_model_dir)

        #Verify TFLite model output with Stable Baselines model output
        verify_tflite(tflite_interpreter, saved_info, show_output=True)

    elif mode == 'deploy':

        #Load the model
        tflite_interpreter = load_tflite_model(tflite_model_dir)

        #Predict actions using TFlite model and send commands to real bittle
        deploy_on_bittle(tflite_interpreter)

    elif mode == 'manual':
        manually_control_bittle()  