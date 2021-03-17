import numpy as np
import sys
import importlib

if importlib.util.find_spec('fastText') != None:
    import fastText
else:
    import fasttext as fastText

EPSILON = 1e-10  # only to prevent division by zero


class FasttextEmbeddingModel:
    def __init__(self, filename):
        self.fmodel = self.fmodel = fastText.load_model(filename)

    def get_class_vector(self, text_value, norm=True):
        vector = self.fmodel.get_word_vector(text_value)
        if norm:
            vector /= (np.linalg.norm(vector) + EPSILON)
        return vector

    def get_instance_vector(self, text_value, norm=True):
        vector = self.fmodel.get_word_vector(text_value)
        if norm:
            vector /= (np.linalg.norm(vector) + EPSILON)
        return vector
