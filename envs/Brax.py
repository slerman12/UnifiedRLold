#@title Colab setup and imports
#@markdown ### ⚠️ PLEASE NOTE:
#@markdown Brax and PyTorch can share GPUs but not TPUs.  To access a GPU runtime: from the Colab menu choose Runtime > Change Runtime Type, then select **'GPU'** in the dropdown.
#@markdown
#@markdown Using TPU is OK too, but then PyTorch should run on CPU.

import os
from functools import partial

import torch

from brax.envs.to_torch import JaxToTorchWrapper
from brax.envs import _envs, create_gym_env

from .shared import *

if 'COLAB_TPU_ADDR' in os.environ:
    from jax.tools import colab_tpu
    colab_tpu.setup_tpu()

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    # BUG: (@lebrice): Getting a weird "CUDA error: out of memory" RuntimeError
    # during JIT, which can be "fixed" by first creating a dummy cuda tensor!
    v = torch.ones(1, device="cuda")


for env_name, env_class in _envs.items():
    env_id = f"brax_{env_name}-v0"
    entry_point = partial(create_gym_env, env_name=env_name)
    if env_id not in gym.envs.registry.env_specs:
        print(f"Registring brax's '{env_name}' env under id '{env_id}'.")
        gym.register(env_id, entry_point=entry_point)


def make(env_name, frame_stack, action_repeat, seed):
    env = gym.make(env_name)
    env = JaxToTorchWrapper(env)
    env = DMEnvFromGym(env)
    return env
