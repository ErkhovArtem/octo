import os

save_dir = "/home/hyperdog/octo/checkpoints/shampoo_100_force"
os.makedirs(save_dir, exist_ok=True)

wandb_experiment_name = "shampoo_100_force"
dataset_name = "grasping_dataset:1.0.0"
# standardize_fn = "select_gripper_position_and_force"
# standardize_fn = "select_gripper_position"
standardize_fn = "select_force"

# defaul parameters
pretrained_path =  "hf://rail-berkeley/octo-small-1.5" #"/home/hyperdog/octo/checkpoints/..." # 
data_dir = "/home/hyperdog/tensorflow_datasets"
batch_size = 32
freeze_transformer = False
