import numpy as np
from motion_imitation.utilities import pose3d
from pybullet_utils  import transformations

URDF_FILENAME = "../models/bittle.urdf"

REF_POS_SCALE = 2.1
INIT_POS = np.array([0, 0, .9])
INIT_ROT = transformations.quaternion_from_euler(0,0, -1.5708)

SIM_TOE_JOINT_IDS = [
    3, # left hand
    5, # left foot
    9, # right hand
    11 # right foot
]
SIM_HIP_JOINT_IDS = [2, 4, 8, 10]

#offset the x,y,z of the torso
SIM_ROOT_OFFSET = np.array([0, 0, .01]) #move in +z direction by .01
#offset the x,y,z of each lower leg joint from the mocap 
SIM_TOE_OFFSET_LOCAL = [
    np.array([-.5, 0.0, 0.01]), #back leg offset back
    np.array([0, 0.0, 0.01]),   
    np.array([-.5, 0.0, 0.01]), #back leg offset back
    np.array([0, 0.0, 0.01])
]


DEFAULT_JOINT_POSE = np.array([0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52])
#doesnt seem to do anything
JOINT_DAMPING = [0.5, 0.05, 0.01,
                 0.5, 0.05, 0.01,
                 0.5, 0.05, 0.01,
                 0.5, 0.05, 0.01]

#the pitch of the robot
FORWARD_DIR_OFFSET = np.array([0, 0, .025])
