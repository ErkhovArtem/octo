from time import sleep
from time import time
from datetime import datetime
import numpy as np
import rtde_control
import rtde_receive
import robotiq_gripper
import sys
import keyboard 
from octo.model.octo_model import OctoModel
import jax
from PIL import Image
import jax.numpy as jnp
import teleop_lib as lib
import cv2
import copy
from scipy.spatial.transform import Rotation as R

def base():
    # Go to base pose
    gripper.move(5, 255, 10)
    sleep(0.5)
    rtde_c.servoJ([-1.5807374159442347,
    -0.9485982100116175,
    1.6254310607910156,
    2.7482194900512695,
    -1.6201594511615198,
    0.04921172186732292], velocity, acceleration, dt, lookahead_time, gain)
    

def get_image(cap, resize):
    ret, image = cap.read()
    if not ret:
        print('Can not read')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = jnp.array(Image.fromarray(image).resize((resize, resize), Image.Resampling.LANCZOS))
    return image

def quaternion_multiply(quaternion1, quaternion0, reverse = False):
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1

    return np.array([x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                     -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0], dtype=np.float64)

def pause():
    timeout = False
    base_pose = False
    while(1):
        try: 
            if keyboard.is_pressed('e'):
                return -1
        except: 
            pass

        try: 
            if keyboard.is_pressed('b'):
                print('---------------------------------')
                print('Move to base pose...')
                base()
                base_pose = True
                timeout = True
        except: 
            pass
            
        try: 
            if keyboard.is_pressed('r'):
                print('---------------------------------')
                print('Program running...')
                if base_pose:
                    return 1
                else:
                    return 0
        except: 
            pass

        if timeout:
            timeout = False
            sleep(0.1)

# connect to robot
try:
    rtde_r = rtde_receive.RTDEReceiveInterface("192.168.88.40")
    rtde_c = rtde_control.RTDEControlInterface("192.168.88.40")
    gripper = robotiq_gripper.RobotiqGripper()
except:
    print('Could not connect to robot!')
    sys.exit(1) # ToDo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

gripper.connect("192.168.88.40", 63352)
if not gripper.is_active():
    try:
        gripper.activate()
    except TimeoutError:
        pass

#  camera init
cap_main = cv2.VideoCapture(0)
if not cap_main.isOpened():
    print("Cannot open camera")
    exit()

cap_wrist = cv2.VideoCapture(2)
if not cap_wrist.isOpened():
    print("Cannot open camera")
    exit()

# Load model
model = OctoModel.load_pretrained("/home/erkhovaa/octo/model_checkpoints/octo_small_delta_eef/eef_50k", 49999)
# model = OctoModel.load_pretrained("/home/erkhovaa/octo/model_checkpoints/octo_small_delta_eef/eef_100k", 49999)

task_texts = ["Pick up the gray cylinder."]
task = model.create_tasks(texts=task_texts)

# UR3 control settings
velocity = 0.5
acceleration = 0.5
dt = 1.0/500  # 2ms
lookahead_time = 0.2
gain = 100

# go to the base pose
base()

previous_image_main = None
previous_image_wrist = None
previous_state = None

sleep(0.5)
print('Starting teleoperation...')

# loop frequncy calculation
start = time()
count = 0

while(1):
    count+=1
            
    target_pose = np.empty(7)

    image_main = get_image(cap_main, 256)
    image_wrist = get_image(cap_wrist, 128)

    tcp_pose = rtde_r.getActualTCPPose()

    if previous_image_main is None:
        previous_image_main = copy.copy(image_main)
    if previous_image_wrist is None:
        previous_image_wrist = copy.copy(image_wrist)

    image_main_stacked = jnp.stack([previous_image_main, image_main], axis=0)
    image_main_stacked = image_main_stacked[jnp.newaxis, ...]
    image_wrist_stacked = jnp.stack([previous_image_wrist, image_wrist], axis=0)
    image_wrist_stacked = image_wrist_stacked[jnp.newaxis, ...]

    observation = {"image_primary": image_main_stacked,
                    "image_wrist": image_wrist_stacked,
                "timestep_pad_mask": jnp.array([[True, True]]),
                "timestep": jnp.array([[0, 1]]),
                    "pad_mask_dict": {
                        "timestep": jnp.array([[True, True]]),  # Shape: (1, 2)
                        "image_primary": jnp.array([[True, True]]),  # Shape: (1, 2)
                        "image_wrist": jnp.array([[True, True]]),  # Shape: (1, 2)
                    },
                    "task_completed": jnp.zeros((1, 2, 4))}  # Shape: (1, 2, 4)

    action = model.sample_actions(observation, task, rng=jax.random.PRNGKey(0),
                                unnormalization_statistics=model.dataset_statistics["action"],)[0][0]
    target_pose[:3] = np.array(tcp_pose[:3]) + action[:3]
    delta_q = R.from_euler('zxy', action[3:6]).as_quat()
    qw = R.from_rotvec(tcp_pose[3:]).as_quat()
    target_pose[3:6] = R.from_quat(quaternion_multiply(qw, delta_q)).as_rotvec()
    target_pose[6] = round(action[6]) if count > 20 else 0

    if rtde_c.getInverseKinematicsHasSolution(target_pose[:-1]):
        q = rtde_c.getInverseKinematics(target_pose[:-1])
        if lib.isInJointLimits(q):
            rtde_c.servoJ(q, velocity, acceleration, dt, lookahead_time, gain)

    gripper_pos = 150 if target_pose[6] else 10
    if (gripper_pos < gripper._max_position - 5) and (gripper_pos > gripper._min_position):
        gripper.move(gripper_pos, 255, 10)
        sleep(0.01)

    previous_image_main = copy.copy(image_main)
    previous_image_wrist = copy.copy(image_wrist)

    try:  # used try so that if user pressed other than the given key error will not be shown 
        if keyboard.is_pressed('s'):
            print('Frequency', count / (time() - start))
            print('---------------------------------')
            print('Program stopped')
            
            cmd = pause()
            
            if cmd == -1:
                break  # finishing the loop
            if cmd == 1:
                previous_image_main = None
                previous_image_wrist = None
            count = 0
            start = time()
    except: 
        pass

cap_main.release()
cap_wrist.release()

print('Exit program...')




