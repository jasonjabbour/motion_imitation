#!/usr/bin/python3
# -*- coding: UTF-8 -*-

''' Upload the modified OpenCat.ino code to the Bittle Arduino Uno. (Found in third_party/OpenCat) '''

from cmath import pi
from types import new_class
import numpy as np

try:
    from serialMaster.ardSerial import *
except:
    from ardSerial import *

roll_pitch_rollRate_pitchRate = [0,0,0,0]*3
real_joint_angles = [0,0,0,0,0,0,0,0]*3

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
    global roll_pitch_rollRate_pitchRate, real_joint_angles
    try:
        #read serial data from MPU6050
        dataPacket = ser.Read_Line().decode('ISO-8859-1')
        dataPacket = dataPacket.split(',')

        #remove the new line from the last spot
        dataPacket[len(dataPacket) - 1] = dataPacket[len(dataPacket) - 1].strip('\r\n')

        #--- IMU --- 
        #read yaw, pitch, roll, and angular velocity (z,y,x)
        yaw = float(dataPacket[1])
        pitch =  -float(dataPacket[3])
        roll = -float(dataPacket[5])
        # convert from deg/s to rad/s
        angularVelocityZ = float(dataPacket[7]) * pi / 180
        angularVelocityY = float(dataPacket[9]) * pi / 180
        angularVelocityX = -float(dataPacket[11]) * pi / 180
        #save data that are needed
        new_imu_reading = np.array([roll, pitch, angularVelocityX, angularVelocityY])

        # --- Join Angles ---
        #get joint angles 8,9,10,11,12,13,14,15
        joint8_angle = float(dataPacket[13])
        joint9_angle = float(dataPacket[15])
        joint10_angle = float(dataPacket[17])
        joint11_angle = float(dataPacket[19])
        joint12_angle = float(dataPacket[21])
        joint13_angle = float(dataPacket[23])
        joint14_angle = float(dataPacket[25])
        joint15_angle = float(dataPacket[27])
        #order joints '9','13','8','12','10','14','11','15'
        new_joint_angles = np.array([joint9_angle,joint13_angle, joint8_angle, joint12_angle, joint10_angle, joint14_angle, joint11_angle, joint15_angle])
        #convert to radians
        new_joint_angles = np.radians(new_joint_angles)

        #Update History
        # IMU append to top and remove last 4
        roll_pitch_rollRate_pitchRate = np.concatenate((new_imu_reading,roll_pitch_rollRate_pitchRate))
        roll_pitch_rollRate_pitchRate = roll_pitch_rollRate_pitchRate[:12]
        # Joint Angles append to top and remove last 8
        real_joint_angles = np.concatenate((new_joint_angles,real_joint_angles))
        real_joint_angles = real_joint_angles[:24]

    except Exception as e:
        #not decoded successfully
        pass
    
    #if exception just return the previous found
    return real_joint_angles, roll_pitch_rollRate_pitchRate

if __name__ == "__main__":
    initializeCommands()

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
    