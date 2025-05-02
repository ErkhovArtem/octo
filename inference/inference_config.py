import numpy as np

# set proprio input type
proprio_input = "force"

# set up logging
enable_logging = True
experiment_name = "shampoo"

# load model
language_instruction = ["Pick up a shampoo bottle."]
checkpoint_step = 29999
if proprio_input == "force_and_position":
        checkpoint_path = "/home/hyperdog/octo/checkpoints/shampoo_100_force_and_position"     
elif proprio_input == "position":
        checkpoint_path = "/home/hyperdog/octo/checkpoints/shampoo_100_position"
elif proprio_input == "force":
        checkpoint_path = "/home/hyperdog/octo/checkpoints/shampoo_100_force"
else:
        raise ValueError(f"<{proprio_input}> is wrong type of proprio data.")

# env configuration
wrist_camera_id = 0
base_pose = np.array([-0.001450840626851857,
        -1.5725587050067347,
        1.5713553428649902,
        0.0008134841918945312,
        1.5712484121322632,
        3.142502719560732,])
env_config = {"max_joint_rotation": 0.15,
              "max_force": 0.8}