import numpy as np
import sys

class RandomEmbeddingModel:
    def __init__(self, dim):
        self.dim = dim

    def get_features(self, text_value):
        vec = np.random.rand(self.dim)
        return vec
