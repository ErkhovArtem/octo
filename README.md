<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/octo-models/octo/blob/main/examples/01_inference_pretrained.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://octo-models.github.io/)
![](https://github.com/rail-berkeley/octo/workflows/run-debug/badge.svg)
![](https://github.com/rail-berkeley/octo/workflows/pre-commit/badge.svg) -->

## Project Description

This repository extends the [Octo foundation model](https://github.com/octo-models/octo) to enable delicate object grasping using force feedback on UR3 robotic arm equipped with Robotiq 2F-85 gripper. The key innovation is the integration of continuous gripper position control with real-time force sensing through custom sensor-equipped fingers. We collected a dataset with 300 expert demonstrations on 3 object categories (tomato, toothpaste, shampoo) via [Echo Teleoperation System](https://github.com/Eterwait/Echo).

<!-- ## Get Started

Follow the installation instructions, then load a pretrained Octo model! See [examples](examples/) for guides to zero-shot evaluation and finetuning and [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1z0vELj_lX9OWeoMG_WvXnQs43aPOEAhz?usp=sharing)
for an inference example.

```python
from octo.model.octo_model import OctoModel
model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")
print(model.get_pretty_spec())
```

![Octo model](docs/assets/teaser.jpg)

Out of the box, Octo supports multiple RGB camera inputs, can control various robot arms,
and can be instructed via language commands or goal images.
Octo uses a modular attention structure in its transformer backbone, allowing it to be effectively finetuned
to robot setups with new sensory inputs, action spaces, and morphologies, using only a small target domain
dataset and accessible compute budgets. -->


## Installation
```bash
conda create -n octo python=3.10
conda activate octo
pip install -e .
pip install -r requirements.txt
```
For GPU:
```bash
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For TPU
```bash
pip install --upgrade "jax[tpu]==0.4.20" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```
See the [Jax Github page](https://github.com/google/jax) for more details on installing Jax.

Test the installation by finetuning on the debug dataset:
```bash
python scripts/finetune.py --config.pretrained_path=hf://rail-berkeley/octo-small-1.5 --debug
```

<!-- ## Checkpoints

You can find pretrained Octo checkpoints [here](https://huggingface.co/rail-berkeley).
At the moment we provide the following model versions:

| Model                                                         | Inference on 1x NVIDIA 4090 | Size       |
|---------------------------------------------------------------|-----------------------------|------------|
| [Octo-Base](https://huggingface.co/rail-berkeley/octo-base)   | 13 it/sec                   | 93M Params |
| [Octo-Small](https://huggingface.co/rail-berkeley/octo-small) | 17 it/sec                   | 27M Params | -->

## Code Structure

|                     | File                                                    | Description                                                                   |
|---------------------|---------------------------------------------------------|-------------------------------------------------------------------------------|
| Finetuning Loop     | [finetune.py](finetuning/finetune.py)                      | Main finetuning script.                                                       |

| Finetuning Hyperparameters     | [funetune_config.py](finetuning/finetune_config.py)       | Defines main hyperparameters for the finetuning run.                             |

| Inference Loop           | [inference.py](inference/inference.py)                 | Main inference 
script.     |

| Inference Hyperparameters     | [inference_config.py](inference/inference_config.py)       | Defines main hyperparameters for the inference run.
run.                             |

| Envs          | [env.py](inference/envs/env.py)    | Custom robot environment                       |

| Visualization       | [visualizer.ipynb](inference/visualizer.ipynb) | Visualization of force and gripper position.                        |

<!-- ## FAQ
#### What is the `timestep_pad_mask` in the observation dictionary?
The `timestep_pad_mask` indicates which observations should be attended to, which is important when using multiple timesteps of observation history. Octo was trained with a history window size of 2, meaning the model can predict an action using both the current observation and the previous observation. However, at the very beginning of the trajectory, there is no previous observation, so we need to set `timestep_pad_mask=False` at the corresponding index. If you use Octo with a window size of 1, `timestep_pad_mask` should always just be `[True]`, indicating that the one and only observation in the window should be attended to. Note that if you wrap your robot environment with the `HistoryWrapper` (see [gym_wrappers.py](octo/utils/gym_wrappers.py)), the `timestep_pad_mask` key will be added to the observation dictionary for you.
#### What is `pad_mask_dict` in the observation dictionary?
While `timestep_pad_mask` indicates which observations should be attended to on a timestep level, `pad_mask_dict` indicates which elements of the observation should be attended to within a single timestep. For example, for datasets without language labels, `pad_mask_dict["language_instruction"]` is set to `False`. For datasets without a wrist camera, `pad_mask_dict["image_wrist"]` is set to `False`. For convenience, if a key is missing from the observation dict, it is equivalent to setting `pad_mask_dict` to `False` for that key.
#### Does `model.sample_actions([...])` return the full trajectory to solve a task?
Octo was pretrained with an action chunking size of 4, meaning it predicts the next 4 actions at once. You can choose to execute all these actions before sampling new ones, or only execute the first action before sampling new ones (also known as receding horizon control). You can also do something more advanced like [temporal ensembling](octo/utils/gym_wrappers.py). -->

<!-- ## Updates for Version 1.5
- Improved cross-attention between visual and language tokens by repeating language tokens at every timestep in the context window.
- Augmented the language instructions in the data with rephrasings from GPT-3.5.
- Bug fixes:
  - Turned off dropout in the diffusion head due to incompatibility with layer norm.
  - Fixed an off-by-one error with the attention mask.
  - Fixed an issue where different image augmentations did not get fresh random seeds. -->

<!-- ## Citation

```
@inproceedings{octo_2023,
    title={Octo: An Open-Source Generalist Robot Policy},
    author = {{Octo Model Team} and Dibya Ghosh and Homer Walke and Karl Pertsch and Kevin Black and Oier Mees and Sudeep Dasari and Joey Hejna and Charles Xu and Jianlan Luo and Tobias Kreiman and {You Liang} Tan and Pannag Sanketi and Quan Vuong and Ted Xiao and Dorsa Sadigh and Chelsea Finn and Sergey Levine},
    booktitle = {Proceedings of Robotics: Science and Systems},
    address  = {Delft, Netherlands},
    year = {2024},
}
``` -->
