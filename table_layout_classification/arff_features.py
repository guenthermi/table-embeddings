import numpy as np
import sys


class ArffFeatures:
    def __init__(self, filepath):
        self.ID_ATTR = 'ID'
        self.attrs = dict()
        self.entities = list()
        self.id_lookup = dict()
        self._parse_arff_features(filepath)
        self._create_id_lookup()
        self._calculate_factors()
        print('Number of entities in .arff file:', len(self.entities))

    def _parse_arff_features(self, filepath):
        self.attrs = dict()
        self.entities = list()
        f = open(filepath, 'r')
        mode = 'meta_data'
        for line in f:
            if mode == 'meta_data':
                if line.startswith('@attribute'):
                    self._parse_attr_def(line)
                if line.startswith('@data'):
                    mode = 'data'
            elif mode == 'data':
                self._parse_entity(line)

    def _parse_attr_def(self, line):
        _, attr_name, attr_type = line.split()
        self.attrs[len(self.attrs)] = (attr_name, attr_type)

    def _parse_entity(self, line):
        entity = dict()
        values = line.split(',')
        for i, value in enumerate(values):
            entity[self.attrs[i][0]] = value
        self.entities.append(entity)
        return

    def _create_id_lookup(self):
        self.id_lookkup = dict()
        for entity in self.entities:
            entity_id = int(entity[self.ID_ATTR])
            self.id_lookup[entity_id] = entity
        return

    def _calculate_factors(self):
        self.factors = dict()
        for attr_id in self.attrs:
            if (self.attrs[attr_id][0] != self.ID_ATTR) and (self.attrs[attr_id][1] in ('numeric', 'numerical')):
                values = [float(entity[self.attrs[attr_id][0]])
                          for entity in self.entities]
                mu = np.mean(values)
                sig = np.std(values)
                if sig == 0:
                    print('WARNING: sig=0 for attr',
                          self.attrs[attr_id][0], file=sys.stderr)
                self.factors[attr_id] = (mu, sig)
        return

    def get_vector(self, id, zscores=True):
        entity = self.id_lookup[id]
        vec = []
        for attr_id in self.attrs:
            if (self.attrs[attr_id][0] != self.ID_ATTR) and (self.attrs[attr_id][1] in ('numeric', 'numerical')):
                if zscores:
                    v = float(entity[self.attrs[attr_id][0]])
                    if self.factors[attr_id][1] != 0:
                        vec.append((float(entity[self.attrs[attr_id][0]]) -
                                    self.factors[attr_id][0]) / self.factors[attr_id][1])
                else:
                    vec.append(entity[self.attrs[attr_id][0]])
        return np.array(vec, dtype='float32')

    def get_all_vector(self):
        return np.array([self.get_vector(id) for id in range(len(self.entities))])
