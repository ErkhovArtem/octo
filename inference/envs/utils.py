import numpy as np
from pathlib import Path

def normalize(data, metadata):
    """Normalize data using dataset statistics while respecting a mask.
    
    Args:
        data: Input array to normalize
        metadata: Dictionary containing:
            - 'mean': Mean values for normalization
            - 'std': Standard deviation values
            - 'mask' (optional): Boolean mask indicating which elements to normalize
            
    Returns:
        Normalized array where mask is True, original values where False
    """
    mask = metadata.get("mask", np.ones_like(metadata["mean"], dtype=bool))
    return np.where(
        mask,
        (data - metadata["mean"]) / (metadata["std"] + 1e-8),  # Small epsilon to avoid division by zero
        data,
    )

def construct_observation(observation, metadata, proprio_data):
    """Process and normalize observation data based on configuration.
    
    Args:
        observation: Raw observation dictionary containing:
            - 'proprio': Proprioceptive data array
        metadata: Normalization parameters (see normalize())
        proprio_data: Configuration string determining which proprioceptive data to use:
            - "force_and_position": Use last 2 dimensions
            - "position": Use only position (second last dimension)
            - otherwise: Use only force (last dimension)
            
    Returns:
        Processed observation dictionary with normalized proprioceptive data
    """
    # Select relevant proprioceptive data
    if proprio_data == "force_and_position":
        observation['proprio'] = observation['proprio'][..., -2:]
    elif proprio_data == "position":
        observation['proprio'] = observation['proprio'][..., -2][..., None]  # Maintain array dimensionality
    else:
        observation['proprio'] = observation['proprio'][..., -1][..., None]

    # Normalize selected proprioceptive data
    proprio_norm = normalize(observation['proprio'][0], metadata)
    observation['proprio'] = observation['proprio'].at[0].set(proprio_norm)
    return observation

class Logger:
    """Logs proprioceptive data to disk with automatic episode numbering."""
    
    def __init__(self, experiment_name, proprio_data):
        """Initialize logger with empty buffer and create output directory.
        
        Args:
            experiment_name: Name of experiment (used for subdirectory)
            proprio_data: Type of proprioceptive data being logged
        """
        self.log_data = np.zeros((0, 2))  # Buffer for current episode data
        self.log_dir = Path() / "logs" / experiment_name / proprio_data
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, observation):
        """Append current timestep's proprioceptive data to buffer.
        
        Args:
            observation: Dictionary containing 'proprio' array with shape (1, 2, N)
        """
        data = observation['proprio'][0, 1, -2:][None, ...]  # Extract current timestep
        self.log_data = np.concatenate([self.log_data, data])

    def reset(self):
        """Clear current episode buffer."""
        self.log_data = np.zeros((0, 2))

    def save(self):
        """Save current episode data to disk with auto-incremented filename."""
        if len(self.log_data) == 0:
            return  # Skip saving if no data

        # Find next available episode number
        episode_number = 1
        while (self.log_dir / f"episode_{episode_number}.npy").exists():
            episode_number += 1

        # Save and reset buffer
        np.save(self.log_dir / f'episode_{episode_number}.npy', self.log_data)
        self.reset()

class NullLogger:
    """Dummy logger that implements the Logger interface but does nothing."""
    
    def __init__(self, *args, **kwargs):
        """Accept any arguments but take no action."""
        pass

    def log(self, *args, **kwargs):
        """Accept logging calls but take no action."""
        pass

    def save(self, *args, **kwargs):
        """Accept save calls but take no action."""
        pass