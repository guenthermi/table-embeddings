import numpy as np
import importlib

if importlib.util.find_spec('fastText') != None:
    import fastText
else:
    import fasttext as fastText


class FastTextModel:
    def __init__(self, filename):
        self.fmodel = fastText.load_model(filename)

    def get_features(self, text_value):
        vec = self.fmodel.get_word_vector(text_value)
        return vec
