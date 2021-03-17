import numpy as np
import sys

sys.path.insert(0, 'embedding/')

NORMALIZATION = 'individual'  # 'individual', 'once', or 'none'
EPSILON = 1e-5  # only to prevent division by zero

from fasttext_web_table_embeddings import FastTextWebTableModel


class WebTableMultiEmbeddingModel:
    def __init__(self, filenames):
        self.fmodels = []
        for filename in filenames:
            self.fmodels.append(FastTextWebTableModel.load_model(filename))

    def get_features(self, text_value):
        header_vectors = []
        data_vectors = []
        for fmodel in self.fmodels:
            header_vectors.append(fmodel.get_header_vector(text_value))
            data_vectors.append(fmodel.get_data_vector(text_value))
        header_vector = np.concatenate(header_vectors)
        data_vector = np.concatenate(data_vectors)
        if NORMALIZATION == 'individual':
            header_vector /= np.linalg.norm(header_vector) + EPSILON
            data_vector /= np.linalg.norm(data_vector) + EPSILON
            return np.concatenate([header_vector, data_vector])
        elif NORMALIZATION == 'once':
            vec = np.concatenate([header_vector, data_vector])
            vec /= np.linalg.norm(vec) + EPSILON
            return vec
        else:
            return np.concatenate([header_vector, data_vector])
