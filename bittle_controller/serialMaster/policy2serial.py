###########
#!/usr/bin/python3
# -*- coding: UTF-8 -*-

''' Upload the modified OpenCat.ino code to the Bittle Arduino Uno. (Found in third_party/OpenCat) '''

from cmath import pi
import numpy as np
 
# #FIX
try:
    from serialMaster.ardSerial import *
except:
    from ardSerial import *


rpy = [0]*18
LEN_IMU = 8
LEN_TOTAL = LEN_IMU + 0 #Quaterians
previous_time = 0
saved_imu_reading_average = [0]*8

imu_data_direct = {}

def initializeCommands():
    try:
        flushSeialOutput(300)
    except Exception as e:
        logger.info("Exception")
        closeSerialBehavior()
        raise e

def sendCommand(task):
    try:
        token = task[0][0]
        wrapper(task)
        # response = ser.main_engine.read_all()
        # logger.info(f"Response is: {response.decode('ISO-8859-1')}")

    except Exception as e:        
        logger.info("Exception")
        closeSerialBehavior()
        raise e

def endCommands():
    closeSerialBehavior()
    logger.info("finish!")


def getBittleIMUSensorInfo():

    global rpy, previous_time, saved_imu_reading_average

    try:

        previous_imu_reading_average = saved_imu_reading_average

        #current time
        current_time = previous_time

        #read serial data from MPU6050 collecting all data since previous read
        allDataPacket = ser.Read_All().decode('ISO-8859-1').split('\r\n')

        try:
            imu_reading_average = []

            for line in allDataPacket:
            
                #Collect the values in line: Yaw, VALUE, Pitch, VALUE, Roll, VALUE, dZ, VALUE, dY, VALUE, dx, VALUE
                #reading = line.strip().split(',') #[1::2]
                reading = line.strip().split(',') # y value, p value, r value, aax value, aay value, aaz value, time

                if len(reading) == LEN_TOTAL:
                #assert len(reading) == LEN_TOTAL, f"Length of Packet should be: {LEN_TOTAL}"

                    #Convert to floats
                    reading = [float(i) for i in reading]

                    #Convert roll pitch yaw rates to radians
                    reading[3:6] = [i*pi/180 for i in reading[3:6]]

                    #save the last time read
                    current_time = reading[6]/1000

                    #Add reading to collection of readings during this time period
                    imu_reading_average.append(reading)

            #Add up all readings found and average them
            num_readings = len(imu_reading_average)
            imu_reading_average = np.array(imu_reading_average)
            imu_reading_average = np.sum(imu_reading_average, axis=0)/num_readings

            #If no readings found, throw exeption and use the previous imu reading
            assert imu_reading_average.size == LEN_TOTAL, f"Length of Packet should be: {LEN_TOTAL}. Using previous IMU Reading"

            #Save the reading
            saved_imu_reading_average = imu_reading_average
        except Exception as e:
            print('error',e)

            #use previous saved data
            imu_reading_average = previous_imu_reading_average

        dt = current_time - previous_time
        previous_time = current_time

        #--- IMU --- 
        #read yaw, pitch, roll, and angular velocity (z,y,x)
        yaw = -imu_reading_average[0]
        pitch = imu_reading_average[1]
        roll = -imu_reading_average[2]

        new_imu_reading = np.array([roll, pitch, yaw])

        #If reading is very wrong, just keep previous reading
        for i in range(len(new_imu_reading)):
            if abs(new_imu_reading[i]) > 2:
                print("here",new_imu_reading[i], 'replaced with',rpy[i])
                new_imu_reading[i] = rpy[i]

        #Update History
        # IMU append to top and remove last 4
        rpy = np.concatenate((new_imu_reading,rpy))
        rpy = rpy[:18]

    except Exception as e:
        print(e)
        #not decoded successfully
    
    #if exception just return the previous found
    return rpy

if __name__ == "__main__":
    initializeCommands()



#Extra code to be deleted:

    # time.sleep(5)
    # action = [50]*8
    # task = ['i',[9,action[0],13,action[1],8,action[2],12, action[3], 10, action[4],14, action[5],11, action[6],15, action[7]],0]
    # print("sending command")
    # sendCommand(task)
    # print("done")

    # print("---------------------------")
    # time.sleep(.1)
    # task = ['j',2]
    # token = task[0][0]
    # wrapper(task)
    # #dataPacket = ser.Read_Line().decode()
    # response = ser.main_engine.read_all()
    # logger.info(f"Response is: {response.decode('ISO-8859-1')}")


    #getBittleIMUSensorInfo()

    # dataPacket = ser.Read_Line().decode('ISO-8859-1')
    # dataPacket = dataPacket.split(',')

    # #remove the new line from the last spot
    # dataPacket[len(dataPacket) - 1] = dataPacket[len(dataPacket) - 1].strip('\r\n')

    # --- Join Angles ---
    #get joint angles 8,9,10,11,12,13,14,15
    # joint8_angle = float(dataPacket[13])
    # joint9_angle = float(dataPacket[15])
    # joint10_angle = float(dataPacket[17])
    # joint11_angle = float(dataPacket[19])
    # joint12_angle = float(dataPacket[21])
    # joint13_angle = float(dataPacket[23])
    # joint14_angle = float(dataPacket[25])
    # joint15_angle = float(dataPacket[27])
    # #order joints '9','13','8','12','10','14','11','15'
    # new_joint_angles = np.array([joint9_angle,joint13_angle, joint8_angle, joint12_angle, joint10_angle, joint14_angle, joint11_angle, joint15_angle])
    # #convert to radians
    # new_joint_angles = np.radians(new_joint_angles)

    # Joint Angles append to top and remove last 8
    # real_joint_angles = np.concatenate((new_joint_angles,real_joint_angles))
    # real_joint_angles = real_joint_angles[:24]

    #Rotate Coordinates
    # new_yaw = -math.atan2(math.sin(roll)*math.sin(yaw)+math.cos(roll)*math.sin(-pi)*math.cos(yaw),-math.sin(roll)*math.cos(yaw)+math.cos(roll)*math.sin(-pi)*math.sin(yaw))
    # #new_pitch = math.acos(math.cos(roll)*math.cos(-pi+pitch))
    # new_pitch = pitch
    # new_roll = math.atan2(-math.sin(-pi), math.sin(roll)*math.cos(-pi))

    # print("Old", yaw, pitch, roll)
    # print("new", new_yaw, new_pitch, new_roll)


    # converted from deg/s to rad/s
    # yaw_rate = r = imu_reading_average[5] * dt
    # pitch_rate = q = imu_reading_average[4] * dt
    # roll_rate = p = imu_reading_average[3] * dt

    # yaw = imu_reading_average[0]
    # pitch = imu_reading_average[1]
    # roll = imu_reading_average[2]
    # # converted from deg/s to rad/s
    # yaw_rate = r = imu_reading_average[3]
    # pitch_rate = q = imu_reading_average[4]
    # roll_rate = p = imu_reading_average[5]

    #DELETE ------- convert rpy to euler
    # yaw = -imu_reading_average[2]
    # pitch = imu_reading_average[1]
    # roll = -imu_reading_average[0]
    # # converted from deg/s to rad/s
    # yaw_rate = r = imu_reading_average[5]
    # pitch_rate = q = imu_reading_average[4]
    # roll_rate = p = imu_reading_average[3]



    # Wx = p - r*math.sin(pitch)
    # Wy = q*math.cos(roll) + r*math.sin(roll)*math.cos(pitch)
    # Wz = r*math.cos(roll)*math.cos(pitch) - q*math.sin(roll)
    
    #save data that are needed in correct order for observational space
    #new_imu_reading = np.array([roll, pitch, 0,  random.uniform(-.5,.5),  random.uniform(-.5,.5), random.uniform(-.5,.5)])
    #new_imu_reading = np.array([roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate])

    #How many seconds have passed
    # dt = current_time - previous_time
    # previous_time = current_time

    #Calculate rate of roll, pitch and yaw
    # roll_rate = (roll - previous_imu_reading[0])/dt
    # pitch_rate = (pitch - previous_imu_reading[1])/dt
    # yaw_rate = (yaw - previous_imu_reading[2])/dt

    #Convert Gyro rate to euler engle rates
    # roll_rate = p + q*math.sin(roll)*math.tan(pitch) + r*math.cos(roll)*math.tan(pitch)
    # pitch_rate = q*math.cos(roll) - r*math.sin(roll)
    # yaw_rate = q*math.sin(roll)/math.cos(pitch) + r*math.cos(roll)/math.cos(pitch)

    #Get inverse of prientation in quaternions 
    # orientation_inversed = -1*imu_reading_average[6:]
    # angular_velocity = [roll_rate, pitch_rate, yaw_rate]
    # #World frame to local frame
    # relative_velocity, _ = env._pybullet_client.multiplyTransforms(
    #     [0, 0, 0], orientation_inversed, angular_velocity,
    #     env._pybullet_client.getQuaternionFromEuler([0, 0, 0])) #(0, 0, 0, 1)

    # roll_rate = relative_velocity[0]
    # pitch_rate = relative_velocity[1]
    # yaw_rate = relative_velocity[2]

# from cmath import pi
# import numpy as np

# #FIX
# try:
#     from serialMaster.ardSerial import *
# except:
#     from ardSerial import *
    

# roll_pitch_rollRate_pitchRate = [0,0,0,0]*3
# real_joint_angles = [0,0,0,0,0,0,0,0]*3

# def initializeCommands():
#     try:
#         flushSeialOutput(300)
#     except Exception as e:
#         logger.info("Exception")
#         closeSerialBehavior()
#         raise e

# def sendCommand(task):
#     try:
#         token = task[0][0]
#         wrapper(task)
#         # response = ser.main_engine.read_all()
#         # logger.info(f"Response is: {response.decode('ISO-8859-1')}")

#     except Exception as e:        
#         logger.info("Exception")
#         closeSerialBehavior()
#         raise e

# def endCommands():
#     closeSerialBehavior()
#     logger.info("finish!")


# def getBittleIMUSensorInfo():
#     global roll_pitch_rollRate_pitchRate, real_joint_angles
#     try:
#         #read serial data from MPU6050
#         dataPacket = ser.Read_Line().decode('ISO-8859-1')
#         dataPacket = dataPacket.split(',')

#         #remove the new line from the last spot
#         dataPacket[len(dataPacket) - 1] = dataPacket[len(dataPacket) - 1].strip('\r\n')

#         #--- IMU --- 
#         #read yaw, pitch, roll, and angular velocity (z,y,x)
#         yaw = float(dataPacket[1])
#         pitch =  -float(dataPacket[3])
#         roll = -float(dataPacket[5])
#         # convert from deg/s to rad/s
#         angularVelocityZ = float(dataPacket[7]) * pi / 180
#         angularVelocityY = float(dataPacket[9]) * pi / 180
#         angularVelocityX = -float(dataPacket[11]) * pi / 180
#         #save data that are needed
#         new_imu_reading = np.array([roll, pitch, angularVelocityX, angularVelocityY])

#         # --- Join Angles ---
#         #get joint angles 8,9,10,11,12,13,14,15
#         joint8_angle = float(dataPacket[13])
#         joint9_angle = float(dataPacket[15])
#         joint10_angle = float(dataPacket[17])
#         joint11_angle = float(dataPacket[19])
#         joint12_angle = float(dataPacket[21])
#         joint13_angle = float(dataPacket[23])
#         joint14_angle = float(dataPacket[25])
#         joint15_angle = float(dataPacket[27])
#         #order joints '9','13','8','12','10','14','11','15'
#         new_joint_angles = np.array([joint9_angle,joint13_angle, joint8_angle, joint12_angle, joint10_angle, joint14_angle, joint11_angle, joint15_angle])
#         #convert to radians
#         new_joint_angles = np.radians(new_joint_angles)

#         #Update History
#         # IMU append to top and remove last 4
#         roll_pitch_rollRate_pitchRate = np.concatenate((new_imu_reading,roll_pitch_rollRate_pitchRate))
#         roll_pitch_rollRate_pitchRate = roll_pitch_rollRate_pitchRate[:12]
#         # Joint Angles append to top and remove last 8
#         real_joint_angles = np.concatenate((new_joint_angles,real_joint_angles))
#         real_joint_angles = real_joint_angles[:24]

#     except Exception as e:
#         #not decoded successfully
#         pass
    
#     #if exception just return the previous found
#     return real_joint_angles, roll_pitch_rollRate_pitchRate