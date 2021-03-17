import numpy as np
import sys

sys.path.insert(0, 'embedding/')

from fasttext_web_table_embeddings import FastTextWebTableModel

EPSILON = 1e-10  # only to prevent division by zero


class WebTableEmbeddingModel:
    def __init__(self, filename):
        self.fmodel = FastTextWebTableModel.load_model(filename)

    def get_class_vector(self, text_value, norm=True):
        vector = self.fmodel.get_header_vector(text_value)
        if norm:
            vector /= (np.linalg.norm(vector) + EPSILON)
        return vector

    def get_instance_vector(self, text_value, norm=True):
        vector = self.fmodel.get_data_vector(text_value)
        if norm:
            vector /= (np.linalg.norm(vector) + EPSILON)
        return vector
