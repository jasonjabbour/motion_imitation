#!/usr/bin/python3
# -*- coding: UTF-8 -*-

from serialMaster.ardSerial import *


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
