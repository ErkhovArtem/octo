import numpy as np
import time
import copy
import jax.numpy as jnp
from cameras import RealSenseCamera
from PIL import Image
import jax
import os


class Env:
    def __init__(self, robot, device, camera_main, camera_wrist, env_config):
        self.camera_main = camera_main
        self.camera_wrist = camera_wrist
        self.robot = robot
        self.device = device
        self.previous_state = None
        self.env_config = env_config
        
    def step(self, action = None):

        if action is not None:
            self._apply_action(action)

        image_main = self._get_image(self.camera_main, resize = 256)
        image_wrist = self._get_image(self.camera_wrist, resize = 128)
        proprio = self._get_proprio()
        
        if self.previous_state is None:
            self.previous_state = {}
            self.previous_state['image_main'] = copy.copy(image_main)
            self.previous_state['image_wrist'] = copy.copy(image_wrist)
            self.previous_state['proprio'] = copy.copy(proprio)

        image_main_stacked = jnp.stack([self.previous_state['image_main'], image_main], axis=0)
        image_wrist_stacked = jnp.stack([self.previous_state['image_wrist'], image_wrist], axis=0)
        proprio_stacked = jnp.stack([self.previous_state['proprio'], proprio], axis=0)

        self.previous_state['image_main'] = copy.copy(image_main)
        self.previous_state['image_wrist'] = copy.copy(image_wrist)
        self.previous_state['proprio'] = copy.copy(proprio)
        
        observation = {"image_primary": image_main_stacked[jnp.newaxis, ...],
                "image_wrist": image_wrist_stacked[jnp.newaxis, ...],
                "proprio": proprio_stacked[jnp.newaxis, ...],
            "timestep_pad_mask": jnp.array([[True, True]]),
            "timestep": jnp.array([[0, 1]]),
                "pad_mask_dict": {
                    "timestep": jnp.array([[True, True]]),  # Shape: (1, 2)
                    "image_primary": jnp.array([[True, True]]),  # Shape: (1, 2)
                    "image_wrist": jnp.array([[True, True]]),  # Shape: (1, 2)
                    "proprio": jnp.array([[True, True]])  # Shape: (1, 2)
                },
                "task_completed": jnp.zeros((1, 2, 4))}  # Shape: (1, 2, 4)
        return observation, info
    
    def reset(self):
        self.robot.move_to_base_pose()
        time.sleep(1)
        self.previous_state = None
        return self.step()

    def _get_image(self, camera, resize):
        if isinstance(camera, RealSenseCamera):
            image = camera.get_frame(depth = False)
        else:
            image = camera.get_frame()
        image = jnp.array(Image.fromarray(image).resize((resize, resize), Image.Resampling.LANCZOS))
        return image
    
    def _get_proprio(self):
        data_from_echo = self.device.read_pose_rad(dof_count=7, read_force_sensor=True)
        if data_from_echo is None:
            return self.previous_state['proprio'], None
        force = data_from_echo[4][0]/4095
        if force > 0.8:
            raise RuntimeError('Can not read from the wrist camera')
        gripper_pose = self.robot.get_current_gripper_pose()[0]/255
        proprio = jnp.array([gripper_pose, force])
        return proprio
    
    def _apply_action(self, action):

        current_angles = self.robot.get_current_joint_angles()
        target_angles = current_angles + np.clip(action[:6], a_min = -self.env_config['max_joint_rotation'], 
                                                 a_max = self.env_config['max_joint_rotation'])
        if max(abs(action[:6])) > self.env_config['max_joint_rotation']:
            print(f"Action was clipped! Max action is {max(abs(action[:6]))}")

        current_gripper_pose = self.robot.get_current_gripper_pose()
        target_gripper_pose = int(current_gripper_pose + round(action[-1] * 255))
        self.robot.move_to_pose(target_angles, target_gripper_pose)
