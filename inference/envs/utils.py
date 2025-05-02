import numpy as np
from datetime import datetime
from pathlib import Path

def normalize(data, metadata):
        mask = metadata.get("mask", np.ones_like(metadata["mean"], dtype=bool))
        return np.where(
            mask,
            (data - metadata["mean"]) / (metadata["std"] + 1e-8),
            data,
        )

def construct_observation(observation, metadata, proprio_data):
    # select observations that will be passed into model
    if proprio_data == "force_and_position":
        observation['proprio'] = observation['proprio'][..., -2:]
    elif proprio_data == "position":
        observation['proprio'] = observation['proprio'][..., -2][..., None]
    else:
        observation['proprio'] = observation['proprio'][..., -1][..., None]

    # normalize observations using dataset statistics
    proprio_norm = normalize(observation['proprio'][0], metadata)
    observation['proprio'] = observation['proprio'].at[0].set(proprio_norm)
    return observation

class Logger:
    def __init__(self, experiment_name, proprio_data):
        self.log_data = np.zeros((0, 2))
        self.log_dir = Path() / "logs" / experiment_name / proprio_data
        self.log_dir.mkdir(parents=True, exist_ok= True)

    def log(self, observation):
        data = observation['proprio'][0, 1, -2:][None, ...]
        self.log_data = np.concatenate([self.log_data, data])

    def reset(self):
        self.log_data = np.zeros((0, 2))

    def save(self):
        if len(self.log_data) == 0:
            return
        episode_number = 1
        while (self.log_dir / f"episode_{episode_number}.npy").exists():
            episode_number += 1
        filename = f'episode_{episode_number}.npy'
        np.save(self.log_dir / filename, self.log_data)
        self.log_data = np.zeros((0, 2))

class NullLogger:
    def __init__(self, *args, **kwargs): pass
    def log(self, *args, **kwargs): pass
    def save(self, *args, **kwargs): pass