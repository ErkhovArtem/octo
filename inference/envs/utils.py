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

def normalize_proprio(observation, metadata):

    proprio_norm = normalize(observation['proprio'][0], metadata)
    observation['proprio'] = observation['proprio'].at[0].set(proprio_norm)
    return observation

class Logger:
    def __init__(self, experiment_name):
        self.log_data = []
        now = datetime.now()
        experiment_name += f'_{now.month}_{now.day}_{now.hour}{now.minute}{now.second}'
        self.log_dir = Path() / "logs" / experiment_name
        self.log_dir.mkdir()

    def log(self, observation):
        data = observation['proprio'][0, 1, 1]
        self.log_data.append(data)

    def save(self):
        episode_number = 1
        while (self.log_dir / f"episode_{episode_number}.npy").exists():
            episode_number += 1
        filename = f'episode_{episode_number}.npy'
        np.save(self.log_dir / filename, self.log_data)
        self.log_data = []