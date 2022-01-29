#!/usr/bin/python3
# -*- coding: UTF-8 -*-

''' Upload the modified OpenCat code to Bittle Arduino Uno. (Found in third_party/OpenCat) '''

from cmath import pi
from serialMaster.ardSerial import *

roll_pitch_rollRate_pitchRate = [0,0,0,0]*3

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
        # logger.info(f"Response is: {response.decode('utf-8')}")

    except Exception as e:        
        logger.info("Exception")
        closeSerialBehavior()
        raise e

def endCommands():
    closeSerialBehavior()
    logger.info("finish!")


def getBittleSensorInfo():
    global roll_pitch_rollRate_pitchRate
    try:
        #read serial data from MPU6050
        dataPacket = ser.Read_Line().decode()
        dataPacket = dataPacket.split(',')

        #remove the new line from the last spot
        dataPacket[len(dataPacket) - 1] = dataPacket[len(dataPacket) - 1].strip('\r\n')

        #read yaw, pitch, roll, and angular velocity (z,y,x)
        yaw = float(dataPacket[1])
        pitch =  -float(dataPacket[3])
        roll = -float(dataPacket[5])
        # convert from deg/s to rad/s
        angularVelocityZ = float(dataPacket[7]) * pi / 180
        angularVelocityY = float(dataPacket[9]) * pi / 180
        angularVelocityX = -float(dataPacket[11]) * pi / 180

        #save data that are needed
        new_reading = [roll, pitch, angularVelocityX, angularVelocityY]

        #append to top and remove last 4
        roll_pitch_rollRate_pitchRate = new_reading + roll_pitch_rollRate_pitchRate 
        roll_pitch_rollRate_pitchRate = roll_pitch_rollRate_pitchRate[:12]

    except Exception as e:
        pass
    
    #if exception just return the previous found
    return roll_pitch_rollRate_pitchRate 

if __name__ == "__main__":
    initializeCommands()
    getBittleSensorInfo()
    