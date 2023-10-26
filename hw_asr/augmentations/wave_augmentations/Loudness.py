import torch_audiomentations
from torch import Tensor
import random
import numpy as np

from hw_asr.augmentations.base import AugmentationBase

class Loudness(AugmentationBase):
    def __init__(self, p, min_top, max_top, *args, **kwargs):
        self.min_top = min_top
        self.max_top = max_top
        self.p = p

    def __call__(self, data: Tensor):
        if random.random() > self.p:
            return data
        wave_norm = data / data.max()
        loudness_scale = np.random.uniform(low=self.min_top, high=self.max_top)
        return wave_norm * loudness_scale