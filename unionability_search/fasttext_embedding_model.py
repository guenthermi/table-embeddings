import numpy as np
import sys
import importlib

if importlib.util.find_spec('fastText') != None:
    import fastText
else:
    import fasttext as fastText

from statistics import mean_vector_similarity

EPSILON = 1e-10  # only to prevent division by zero


class FasttextEmbeddingModel:
    def __init__(self, filename):
        self.fmodel = fastText.load_model(filename)

    def get_vector(self, text_value, norm=False):
        vector = self.fmodel.get_word_vector(text_value)
        if np.linalg.norm(vector) > 0:
            if norm:
                vector /= (np.linalg.norm(vector) + EPSILON)
            return vector
        else:
            return None

    def get_approximated_unionability_score(self, col1, col2, header1, header2,
                                            model_headers=True):
        a, b = [], []
        h_a, h_b = [], []
        for elem in col1:
            v = self.get_vector(elem)
            if v is not None:
                a.append(v)

        if model_headers:
            v = self.get_vector(header1)
            if v is not None:
                h_a.append(v)

        for elem in col2:
            v = self.get_vector(elem)
            if v is not None:
                b.append(v)

        if model_headers:
            v = self.get_vector(header2)
            if v is not None:
                h_b.append(v)

        if model_headers:
            scores = (mean_vector_similarity(np.array(a), np.array(b)),
                      mean_vector_similarity(np.array(a), np.array(h_b)),
                      mean_vector_similarity(np.array(h_a), np.array(b)))
            scores = [x if x is not None else -1 for x in scores]
            return scores
        else:
            score = mean_vector_similarity(np.array(a), np.array(b))
            return score if score is not None else -1
