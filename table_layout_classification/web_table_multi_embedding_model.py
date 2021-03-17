import numpy as np
import sys

sys.path.insert(0, 'embedding/')

from fasttext_web_table_embeddings import FastTextWebTableModel


class WebTableMultiEmbeddingModel:
    def __init__(self, filenames):
        self.fmodels = []
        for filename in filenames:
            self.fmodels.append(FastTextWebTableModel.load_model(filename))

    def get_features(self, column):
        vecs = []
        for term in column:
            vec = []
            for fmodel in self.fmodels:
                vec.append(fmodel.get_header_vector(term))
                vec.append(fmodel.get_data_vector(term))
            vecs.append(np.concatenate(vec))
        return vecs
