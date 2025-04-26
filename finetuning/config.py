from typing import Any, Dict

save_dir = "/home/hyperdog/octo/checkpoints/tomato_100"
experiment_name = "tomato_100"
dataset_name = "grasping_dataset:1.0.1"

# defaul parameters
pretrained_path =  "hf://rail-berkeley/octo-small-1.5" #"/home/hyperdog/octo/checkpoints/grasping_60" # 
data_dir = "/home/hyperdog/tensorflow_datasets"
batch_size = 32
freeze_transformer = False