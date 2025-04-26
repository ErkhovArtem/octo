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
    def __init__(self):
        self.log_data = []
        self.path = Path() / "logs"
        self.path.mkdir(exist_ok=True)

    def log(self, observation):
        data = observation['proprio'][0, 1, 1]
        self.log_data.append(data)

    def save(self):
        now = datetime.now()
        filename = f'log_{now.month}_{now.day}_{now.hour}{now.minute}{now.second}.npy'
        np.save(self.path / filename, self.log_data)
        self.log_data = []