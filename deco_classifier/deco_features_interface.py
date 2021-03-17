import sys
import json
import pandas as pd
import numpy as np
from collections import defaultdict


class DECOFeaturesInterface:
    """Provides an interface for the DECO features file. It requires the DECO
    features file and a json file with a description of the features as input.
    """

    def __init__(self, features_filename, features_description_file, sheet_set=None):
        # Load the DECO feature csv file into a pandas dataframe
        print('Load deco features csv file ...')
        self.feature_df = self._load_feature_file(features_filename)
        # Load feature descriptions (types, value ranges)
        print('Load deco feature description file ...')
        self.feature_types, self.feature_ranges = self._load_feature_description(
            features_description_file)
        # build a datastructure that is useful for feature lookups
        print('Create deco feature vectors ...')
        self.features = self._create_feature_vectors(sheet_set)
        # get feature dimensionality
        for sheet_key in self.features:
            for vector in self.features[sheet_key].values():
                self.dim = len(vector)
                break
            break
        return

    def get_feature_vector(self, filename, sheetname, col, row, default='zero'):
        """Returns feature vector for cell.
        """
        if (filename, sheetname) in self.features:
            if (col, row) in self.features[(filename, sheetname)]:
                return self.features[(filename, sheetname)][(col, row)]
        print('ERROR: can not obtain a feature vector for',
              (col, row), 'in', (filename, sheetname))
        if default == 'zero':
            return [0] * self.dim
        return None

    def _construct_feature_vector(self, row):
        features = []
        for i, (feature_name, feature_type) in enumerate(self.feature_types):
            feature = row[i]
            if feature_type == 'boolean':
                features.append(int(feature))
            elif feature_type == 'numeric':
                features.append(feature)
            elif feature_type == 'categorical':
                vector = [0] * self.feature_ranges[feature_name]
                if feature > self.feature_ranges[feature_name]:
                    print('ERROR: feature', feature_name, 'out of range.')
                    return None
                vector[feature] = 1
                features += vector
            else:
                print('ERROR: Unknown feature type', feature_type)
                return None
        return np.array(features)

    def _load_feature_file(self, filename):
        return pd.read_csv(filename)

    def _load_feature_description(self, filename):
        f = open(filename, 'r')
        data = json.load(f)
        f.close()
        return list(data['types'].items()), data['ranges']

    def _create_feature_vectors(self, sheet_set):
        features = defaultdict(dict)
        size = len(self.feature_df.index)
        selection = [x for (x, y) in self.feature_types]  # in feature descr.
        iterator = self.feature_df.iterrows()
        if sheet_set is not None:
            valid_files = set([x for (x, y) in sheet_set])
            valid_sheets = set([y for (x, y) in sheet_set])
            iterator = self.feature_df[self.feature_df['file_name'].isin(
                valid_files) & self.feature_df['sheet_name'].isin(valid_sheets)].iterrows()
        for index, row in iterator:
            sheet_key = (row['file_name'], row['sheet_name'])
            cell_keys = []
            if (row['orign_min_col'] != row['orign_max_col']) or (
                    row['orign_min_row'] != row['orign_max_row']):
                # construct multiple cell keys for merged cells
                for i in range(row['orign_min_col'] - 1, row['orign_max_col']):
                    for j in range(row['orign_min_row'] - 1, row['orign_max_row']):
                        cell_keys.append((i, j))
            else:
                cell_keys = [(int(row['orign_min_col']) - 1,
                              int(row['orign_min_row']) - 1)]
            feature_vector = self._construct_feature_vector(
                list(row[selection]))
            if feature_vector is not None:
                for cell_key in cell_keys:
                    features[sheet_key][cell_key] = feature_vector
            else:
                print('ERROR can not obtain feature vector for cell', cell_keys,
                      'in sheet', row['sheet_name'], 'in file', row['file_name'])
            if index % 1000 == 0:
                print('\rLoad Features %05.2f %%' %
                      (float(index) * 100 / size), end='')
        print('')
        return features
