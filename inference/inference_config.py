import numpy as np

language_instruction = ["Pick up a tomato."]
object = "tomato"
wrist_camera_id = 0
base_pose = np.array([-0.001450840626851857,
        -1.5725587050067347,
        1.5713553428649902,
        0.0008134841918945312,
        1.5712484121322632,
        3.142502719560732,])
env_config = {"max_joint_rotation": 0.15}

use_forse = True
checkpoint_step = 19999
if use_forse:
        checkpoint_path = "/home/hyperdog/octo/checkpoints/tomato_100"
        experiment_name = object + "_forcefull"
else:
        checkpoint_path = "/home/hyperdog/octo/checkpoints/tomato_100_no_force"
        experiment_name = object + "_forceless"
