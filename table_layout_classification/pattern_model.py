import numpy as np
import string


class PatternModel:
    def __init__(self, pattern_size=7):
        self.pattern_size = pattern_size
        self.content_types = 'NAPSX'

    def get_features(self, column):
        patterns = self.get_patterns(column)
        features = []
        for pattern in patterns:
            features.append(self._pattern2vector(pattern))
        return features

    def get_patterns(self, column):
        patterns = []
        for cell in column:
            patterns.append(self._get_content_pattern(cell))
        return patterns

    def get_features_dim(self):
        return self.pattern_size * len(self.content_types)

    def _pattern2vector(self, pattern):
        vec = []
        for t in self.content_types:
            for p in pattern:
                vec.append(1 if t == p else -1)
        vec = np.array(vec, dtype='float32')
        vec_len = np.linalg.norm(vec)
        if vec_len > 0:
            vec = vec/vec_len
        else:
            vec = zeros(len(vec))
        return vec

    def _get_content_type(self, c):
        if c in [str(x) for x in range(10)]:
            return 'N'
        if c in string.ascii_letters:
            return 'A'
        if c in '.!;():?,\\-\'"':
            return 'P'
        return 'S'

    def _get_content_pattern(self, text):
        clean_text = ''.join(text.split())
        pattern = ['X'] * self.pattern_size
        for i, c in enumerate(clean_text):
            if i >= self.pattern_size:
                break
            pattern[i] = self._get_content_type(c)
        return pattern
