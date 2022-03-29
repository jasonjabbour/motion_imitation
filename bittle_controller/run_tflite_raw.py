
import tflite_runtime.interpreter as tflite
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import argparse

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

        #Send action:

        #Receive Observations:

def model_processing_time(plot_time=False, trials=5):
    '''#Run 5 Different Saved Info Files 5 Times
        Each info file consists of two episodes with 500 steps each'''
    
    trial_process_times_lst = []

    #Run 5 Different Saved Info Files 5 Times
    for i in range(trials):
        for i in range(trials):
            #Get saved info to pass into model
            file_number = i + 1
            saved_info = get_saved_info('data/saved_info_2ep_1000steps' + str(file_number))

            #Load the model
            tflite_interpreter = load_model(args.tflite_model)

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


def load_model(tflite_model_name):
    '''Load and return a TFLite model
    
    tflite_model_name: tflite file name
    '''
    #TFlite model path
    model_save_file = tflite_model_name + ".tflite"

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
    arg_parser.add_argument("--tflite_model", dest="tflite_model", type=str, default='output/bittle_frozen_axis1')
    arg_parser.add_argument("--verification_info", dest="verification_info", type=str, default='data/saved_info_1ep')

    args = arg_parser.parse_args()

    #Frozen TFLite Model Directory
    tflite_model = args.tflite_model
    mode = args.mode

    if mode == 'time':
        #Plot the timing results 
        plot_time_answer = False
        plot_time_answer = input("Would you like to plot the time for each trial? [Y/N] ")
        if plot_time_answer.lower() == "yes" or plot_time_answer.lower() == "y":
            plot_time_answer = True

        #Time the processing time of the tflite model
        model_processing_time(plot_time_answer)

    elif mode == 'verify':
        #Get saved verification data
        saved_info = get_saved_info(args.verification_info)

        #Load the model
        tflite_interpreter = load_model(args.tflite_model)

        #Verify TFLite model output with Stable Baselines model output
        verify_tflite(tflite_interpreter, saved_info, show_output=True)






        

