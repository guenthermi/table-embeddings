
import time
import numpy as np

EPSILON = 1e-10  # only to prevent division by zero


class Word2VecModel:
    def __init__(self, filename, model_format='word2vec'):
        self.terms, self.vectors = self._parse_vectors(filename, model_format)
        self.terms_tree = self._build_term_tree()

    def get_class_vector(self, text_value, norm=True):
        vector = self._get_vector(text_value)
        if norm:
            vector /= (np.linalg.norm(vector) + EPSILON)
        return vector

    def get_instance_vector(self, text_value, norm=True):
        vector = self._get_vector(text_value)
        if norm:
            vector /= (np.linalg.norm(vector) + EPSILON)
        return vector

    def _get_vector(self, text_value, tokenization_strategy='simple'):
        if (text_value is None) or (len(text_value) == 0):
            print('ERROR: Received empty text value')
            quit()
            return np.zeros(self.vectors[0].shape[0])
        splits = text_value.replace(' ', '_').split('_')
        i = 1
        j = 0
        current = [self.terms_tree, None, -1]
        vector = None
        last_match = (0, None, -1)
        count = 0
        while (i <= len(splits)) or (type(last_match[1]) != type(None)):
            subword = '_'.join(splits[j:i])
            if subword in current[0]:
                current = current[0][subword]
                if current[1] is not None:
                    last_match = (i, current[1], current[2])
            else:
                if type(last_match[1]) != type(None):
                    if type(vector) != type(None):
                        if tokenization_strategy == 'log10':
                            vector += last_match[1] * \
                                np.log10(last_match[2])
                            count += np.log10(last_match[2])
                        else:  # 'simple' or different
                            vector += last_match[1]
                            count += 1
                    else:
                        if tokenization_strategy == 'log10':
                            vector = last_match[1] * \
                                np.log10(last_match[2])
                            count += np.log10(last_match[2])
                        else:  # 'simple' or different
                            vector = last_match[1]
                            count += 1
                    j = last_match[0]
                    i = j
                    last_match = (0, None, -1)
                else:
                    j += 1
                    i = j
                current = [self.terms_tree, None, -1]
            i += 1
        if type(vector) != type(None):
            vector /= count
            return vector
        else:
            return np.zeros(self.vectors[0].shape[0])

    def _parse_vectors(self, filename, model_format):
        terms = []
        vectors = []
        f = open(filename, 'r')
        if model_format == 'word2vec':
            f.readline()
        t = time.time()
        for i, line in enumerate(f):
            if i > 0 and i % 100000 == 0:
                print('\rParsed', i, 'vectors %05.2fs/100K vectors' %
                      (time.time() - t,), end='')
                t = time.time()
            splits = line.split(' ')
            if len(splits) != 301:
                print('ERROR while parsing line:', line)
                continue
            else:
                terms.append(splits[0])
                vectors.append(np.array(splits[1:], dtype='float32'))
        print()
        return terms, vectors

    def _build_term_tree(self):
        term_dict = dict()
        for (term, vector, freq) in zip(self.terms, self.vectors, range(len(self.vectors))):
            splits = term.split('_')
            current = [term_dict, None, -1]
            i = 1
            while i <= len(splits):
                subterm = '_'.join(splits[:i])
                if subterm in current[0]:
                    current = current[0][subterm]
                else:
                    current[0][subterm] = [dict(), None, -1]
                    current = current[0][subterm]
                i += 1
            current[1] = vector
            current[2] = freq
        return term_dict
