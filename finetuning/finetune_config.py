save_dir = "/home/hyperdog/octo/checkpoints/toothpaste_100_no_force"
wandb_experiment_name = "toothpaste_100_no_force"
dataset_name = "grasping_dataset:1.0.2"
use_force = False
if use_force:
    standardize_fn = "select_gripper_position_and_force"
else:
    standardize_fn = "select_gripper_position"

# defaul parameters
pretrained_path =  "hf://rail-berkeley/octo-small-1.5" #"/home/hyperdog/octo/checkpoints/..." # 
data_dir = "/home/hyperdog/tensorflow_datasets"
batch_size = 32
freeze_transformer = False
