import numpy as np

checkpoint_path = "/home/hyperdog/octo/checkpoints/shampoo_100_cleared_data"
checkpoint_step = 39999
# language_instruction = ["Pick up the blue box."]
language_instruction = ["Pick up the orange shampoo bottle."]
wrist_camera_id = 0
base_pose = np.array([-0.001450840626851857,
        -1.5725587050067347,
        1.5713553428649902,
        0.0008134841918945312,
        1.5712484121322632,
        3.142502719560732,])
env_config = {"max_joint_rotation": 0.15}
