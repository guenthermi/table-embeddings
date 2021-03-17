
import random
import re
from whatthelang import WhatTheLang

class DataLoader:

    def __init__(self, corpus_interface, load=True):
        self.FILTER_ENGLISH_TABLES = True
        self.corpus_interface = corpus_interface
        self.table_data = dict()
        self.labels = dict()
        self.load_table_data()

    def load_table_data(self):
        """ Loads table data from the corpus.
        """
        self.table_data = self.corpus_interface.get_table_data()
        self.labels = self.corpus_interface.get_labels()
        if self.FILTER_ENGLISH_TABLES:
            r = re.compile('[0-9]+')
            new_table_data = dict()
            new_labels = dict()
            wtl = WhatTheLang()
            for id in self.table_data:
                text = ' '.join([' '.join(x)for x in self.table_data[id]])
                if wtl.predict_lang(text) == 'en':
                    new_table_data[id] = self.table_data[id]
                    new_labels[id] = self.labels[id]
            self.table_data = new_table_data
            self.labels = new_labels
            print('Number of loaded (English) tables:', len(self.table_data))
        return

    def get_label_set(self):
        """ Returns a set of all possible labels.
        """
        return set(self.labels.values())

    def split_data(self, splitting={'train': 0.4, 'test': 0.1, 'valid': 0.5},
                   shuffle=True, folds=1):
        """Splits table and label data into categories like train, test, and
        validation set.
        """
        # list of all table ids
        id_list = list(self.table_data.keys())

        # shuffle list of table ids
        if shuffle:
            random.shuffle(id_list)

        # split list of table ids according to the configuration in `splitting`
        result = []

        outer_splitting = {
            'train': splitting['train'] + splitting['test'],
            'valid': splitting['valid']
        }

        outer_split = self._get_splitting(outer_splitting, id_list)
        if folds == 1:
            inner_splitting = {
                'train': splitting['train'] / (splitting['train'] + splitting['test']),
                'test': splitting['test'] / (splitting['train'] + splitting['test']),
            }
            inner_split = self._get_splitting(
                inner_splitting, outer_split['train']['table_ids'])
            fold = {
                'train': inner_split['train'],
                'test': inner_split['test'],
                'valid': outer_split['valid']
            }
            result.append(fold)
            return result
        else:
            for i in range(folds):
                inner_splitting = {
                    'train1': i * splitting['test'] / (splitting['train'] + splitting['test']),
                    'test': splitting['test'] / (splitting['train'] + splitting['test']),
                    'train2': (folds - i - 1) * splitting['test'] / (splitting['train'] + splitting['test']),
                }
                inner_split = self._get_splitting(
                    inner_splitting, outer_split['train']['table_ids'])
                fold = {
                    'train': {key: inner_split['train1'][key] + inner_split['train2'][key] for key in inner_split['train1']},
                    'test': inner_split['test'],
                    'valid': outer_split['valid']
                }
                result.append(fold)

            return result

    def _get_splitting(self, splitting, id_list):
        result = dict()
        last_bound = 0
        passed_splitting = 0
        for key in splitting:
            next_splitting = passed_splitting + splitting[key]
            next_bound = int(next_splitting * len(id_list))
            result[key] = {
                'table_ids': [id for id in id_list[last_bound:next_bound]],
                'table_data': [self.table_data[id] for id in id_list[last_bound:next_bound]],
                'labels': [self.labels[id] for id in id_list[last_bound:next_bound]]
            }
            passed_splitting = next_splitting
            last_bound = next_bound
        return result
