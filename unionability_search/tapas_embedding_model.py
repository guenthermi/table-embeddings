import numpy as np
import pandas as pd

from transformers import TapasTokenizer, TapasModel

from statistics import mean_vector_similarity

EPSILON = 1e-10  # only to prevent division by zero


class TapasEmbeddingModel:
    def __init__(self, model_name='google/tapas-base'):
        self.tokenizer = TapasTokenizer.from_pretrained(model_name)
        self.model = TapasModel.from_pretrained(model_name)

    def get_vectors(self, text_values, header='', norm=False):
        df = pd.DataFrame.from_dict(
            {header: [x[:100] for x in text_values[:100]]})
        print(len([x[:100] for x in text_values[:100]]), [x[:100]
                                                          for x in text_values[:100]])
        inputs = self.tokenizer(table=df, return_tensors="pt")
        outputs = self.model(**inputs)

        vector = outputs.last_hidden_state.detach().numpy()[0]
        if np.linalg.norm(vector) > 0:
            if norm:
                vector /= (np.linalg.norm(vector) + EPSILON)
            return vector
        else:
            return None

    def get_approximated_unionability_score(self, col1, col2, header1, header2,
                                            model_headers=True):
        score1, score2, score3 = None, None, None
        if model_headers:
            # a_h = self.get_vectors([], header=header1)
            b_h = self.get_vectors([], header=header2)
            a = self.get_vectors(col1)
            b = self.get_vectors(col2)
            score1 = mean_vector_similarity(np.array(a), np.array(b))
            score2 = mean_vector_similarity(np.array(a), np.array(b_h))
            score3 = 0  # mean_vector_similarity(np.array(a_h), np.array(b))
        else:
            a = self.get_vectors(col1)
            b = self.get_vectors(col2)
            score1 = mean_vector_similarity(np.array(a), np.array(b))
            score2 = score1
            score3 = score1
        return (score1, score2, score3)
