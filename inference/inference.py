from time import sleep
from time import time
import numpy as np
import keyboard 
from octo.model.octo_model import OctoModel
import jax
from inference_config import *
import sys
import os

lib_path = os.path.join(os.path.dirname(__file__), 'envs')
sys.path.append(lib_path)
from cameras import RealSenseCamera, WebCamera
from echo_teleoperation import Echo
from ur_rtde import UR3Teleop
from env import ForcefullEnv, ForcelessEnv
from utils import Logger, normalize_proprio

def pause():
    timeout = False
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
                env.reset()
                logger.save()
                timeout = True
        except: 
            pass
            
        try: 
            if keyboard.is_pressed('r'):
                print('---------------------------------')
                print('Program running...')
                return 0
        except: 
            pass

        if timeout:
            timeout = False
            sleep(0.1)


# Load model
model = OctoModel.load_pretrained(checkpoint_path, checkpoint_step)

robot = UR3Teleop(ip="192.168.1.110", base_pose=base_pose, lookahead_time=0.1, gain=200)

#  camera init
camera_main = RealSenseCamera(capture_frequency=30, width=640, height=480)
camera_wrist = WebCamera(camera_id = wrist_camera_id)

# Init Echo to get force data
device = Echo()

# create env
if use_forse:
    env = ForcefullEnv(robot, device, camera_main, camera_wrist, env_config)
else:
    env = ForcelessEnv(robot, device, camera_main, camera_wrist, env_config)
logger = Logger(experiment_name = experiment_name)

task_texts = language_instruction
task = model.create_tasks(texts=task_texts)

action_proprio_metadata = jax.tree_map(
                lambda x: np.array(x),
                model.dataset_statistics,
                is_leaf=lambda x: isinstance(x, list),
            )

# reset env
observation = env.reset()

print('Starting teleoperation...')

# loop frequncy calculation
start = time()
count = 0

while(1):
    count+=1
    
    action = model.sample_actions(observation, task, rng=jax.random.PRNGKey(0),
                                unnormalization_statistics=model.dataset_statistics["action"],)[0][0]
    observation = env.step(action)
    logger.log(observation)
    observation = normalize_proprio(observation, action_proprio_metadata["proprio"])
    

    try:  # used try so that if user pressed other than the given key error will not be shown 
        if keyboard.is_pressed('s'):
            print('Frequency', count / (time() - start))
            print('---------------------------------')
            print('Program stopped')
            print(count)
            cmd = pause()           
            if cmd == -1:
                break  # finishing the loop
            count = 0
            start = time()
    except: 
        pass

logger.save()
camera_main.release()
camera_wrist.release()

print('Exit program...')




