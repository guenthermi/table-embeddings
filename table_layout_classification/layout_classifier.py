
import os
import glob
import sys
import time
import numpy as np
from collections import defaultdict

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Dot
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.ensemble import RandomForestClassifier

from sklearn.utils import class_weight

CHECKPOINTER_PATH = '/tmp/model%d.bin'


class LateModelCheckpoint(ModelCheckpoint):

    def __init__(self, min_epochs, *args, **kwargs):
        self.min_epochs = min_epochs
        super(LateModelCheckpoint, self).__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        # only save after min epochs
        if epoch > self.min_epochs:
            super(LateModelCheckpoint, self).on_epoch_end(epoch, logs)


class LayoutClassifier:
    def __init__(self, pattern_model, word_embedding_model,
                 web_table_embedding_model, arff_features):

        self.MAX_SEQUENCE_SIZE = 10

        self.models = {
            'rf_model': None,
            'web_table_model': None,
            'pattern_model': None,
            'word_embedding_model': None
        }

        # each model has a set of feature generators that extract features
        self.feature_generators = {}
        if pattern_model:
            self.feature_generators['pattern_model'] = {
                'patterns': pattern_model
            }
        if web_table_embedding_model:
            self.feature_generators['web_table_model'] = {
                'web_table_embeddings': web_table_embedding_model
            }
        if word_embedding_model:
            self.feature_generators['word_embedding_model'] = {
                'word_embedding_model': word_embedding_model
            }

        # structural features extracted from the arff file
        self.global_features = arff_features

    def create_rf_model(self, label_index=None):
        """Creates random forest classifier (key: 'rf_model') for (global)
        structured features.
        """
        model = RandomForestClassifier(
            n_estimators=100, class_weight='balanced')
        if label_index != None:
            if ('rf_model' not in self.models) or (self.models['rf_model'] == None):
                self.models['rf_model'] = dict()
            self.models['rf_model'][label_index] = model
        else:
            self.models['rf_model'] = model

    def create_lstm_model(self, model_name, input_dim,
                          global_features_input_dim, output_dim, label_index=None):
        """
        Create the LSTM network that uses only embedding features.
        """
        print('input_dim', input_dim, 'output_dim', output_dim)

        last_activation = 'sigmoid' if label_index is not None else 'softmax'
        loss = 'binary_crossentropy' if label_index is not None else 'categorical_crossentropy'

        i1 = Input(shape=(10, input_dim))
        i2 = Input(shape=(10, input_dim))
        i3 = Input(shape=(10, input_dim))
        i4 = Input(shape=(10, input_dim))
        input1 = Dropout(0.3)(i1)
        input2 = Dropout(0.3)(i2)
        input3 = Dropout(0.3)(i3)
        input4 = Dropout(0.3)(i4)

        weighting = Dense(10, activation='softmax',
                          kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-2))

        f1 = Flatten()(input1)
        w1 = weighting(f1)
        x1, s1, _ = LSTM(8, dropout=0.5, recurrent_dropout=0.1,
                         return_sequences=True, return_state=True)(input1)
        p1 = Dot(axes=1)([x1, w1])

        f2 = Flatten()(input2)
        w2 = weighting(f2)
        x2, s2, _ = LSTM(8, dropout=0.5, recurrent_dropout=0.1,
                         return_sequences=True, return_state=True)(input2)
        p2 = Dot(axes=1)([x2, w2])

        f3 = Flatten()(input3)
        w3 = weighting(f3)
        x3, s3, _ = LSTM(8, dropout=0.5, recurrent_dropout=0.1,
                         return_sequences=True, return_state=True)(input3)
        p3 = Dot(axes=1)([x3, w3])

        f4 = Flatten()(input4)
        w4 = weighting(f4)
        x4, s4, _ = LSTM(8, dropout=0.5, recurrent_dropout=0.1,
                         return_sequences=True, return_state=True)(input4)
        p4 = Dot(axes=1)([x4, w4])

        c = Concatenate(axis=1)([p1, p2, p3, p4, s1, s2, s3, s4])
        d1 = Dense(100, activation='sigmoid',
                   kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(c)
        d1 = Dropout(0.3)(d1)
        last_d = d1
        for i in range(0):
            d_1 = Dense(50, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(
                l1=1e-5, l2=1e-4))(last_d)
            d_2 = Dense(50, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(
                l1=1e-5, l2=1e-4))(d_1)
            d_3 = Dense(50, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(
                l1=1e-5, l2=1e-4))(d_2)
            d_4 = Dense(100, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(
                l1=1e-5, l2=1e-4))(d_3)
            d_4 = Dropout(0.1)(d_4)
            dadd = Add()([last_d, d_4])
            last_d = dadd
        out = Dense(output_dim, activation=last_activation,
                    input_dim=100)(last_d)

        model = Model(
            inputs=[i1, i2, i3, i4], outputs=out)
        model.compile(loss=loss, optimizer=Adam(
            learning_rate=1e-3), metrics=['accuracy'])
        if label_index == None:
            self.models[model_name] = model
        else:
            if (model_name not in self.models) or (self.models[model_name] == None):
                self.models[model_name] = dict()
            self.models[model_name][label_index] = model

    def train_lstm_model_with_masking(self, model_name, feature_dict,
                                      label_dict, multi_model=False, epochs=20):
        """ Trains an LSTM network with feature sequences of a fixed length.
        To obtain feature sequences with equal length masking has to be enabled
        in `_get_table_feature_vector_for_lstm()`.
        """
        features = feature_dict['train']
        labels = label_dict['train']
        transformed_features = [np.array(a) for a in zip(*features)]

        valid_features = feature_dict['valid']
        valid_labels = label_dict['valid']
        valid_transformed_features = [
            np.array(a) for a in zip(*valid_features)]
        print('TODO Global Feature Shape', np.array(transformed_features[:-1]).shape) # TODO check if this is right
        if multi_model:
            for i in self.models[model_name]:
                if self.models[model_name][i] == None:
                    print('ERROR: Currently, no model is created.', file=sys.stderr)

                class_weights = class_weight.compute_class_weight('balanced',
                                                                  np.unique(
                                                                      labels.T[i]),
                                                                  labels.T[i])
                class_weights = {i: class_weights[i]
                                 for i in range(len(class_weights))}
                print('Class weights:', class_weights)
                model_save_path = CHECKPOINTER_PATH % (int(time.time()),)
                checkpointer = LateModelCheckpoint(
                    25, filepath=model_save_path, verbose=1, save_best_only=True, monitor='loss', save_weights_only=True)
                self.models[model_name][i].fit(transformed_features[:-1], labels.T[i], epochs=epochs,
                                               batch_size=32, validation_data=(
                    valid_transformed_features[:-1],
                    valid_labels.T[i]), class_weight=class_weights, callbacks=[checkpointer])
                self.models[model_name][i].load_weights(model_save_path)
                for f in glob.glob(model_save_path + '*'):
                    os.remove(f)
        else:
            if self.models[model_name] == None:
                print('ERROR: Currently, no model is created.', file=sys.stderr)

            class_weights = class_weight.compute_class_weight('balanced',
                                                              np.unique(
                                                                  [np.argmax(x) for x in labels]),
                                                              [np.argmax(x) for x in labels])
            sample_weights = np.array(
                [class_weights[np.argmax(x)] for x in labels])

            self.models[model_name].fit(transformed_features[:-1], labels, epochs=epochs,
                                        batch_size=32, validation_data=(
                                            valid_transformed_features[:-1],
                                            valid_labels), sample_weight=sample_weights)
        return

    def train_rf_model(self, feature_dict, label_dict, multi_model=False, use_test_set=False):
        """ Trains a random forest classifier (key: 'rf_model') by using only
        the (global) structured features.
        """
        transformed_features = [a for a in zip(*feature_dict['train'])][-1]
        if use_test_set:
            transformed_features + [a for a in zip(*feature_dict['test'])][-1]
        if multi_model:
            for i in self.models['rf_model']:
                self.models['rf_model'][i].fit(
                    transformed_features,
                    label_dict['train'].T[i])

        else:
            self.models['rf_model'].fit(
                transformed_features,
                [list(x).index(1) for x in label_dict['train']])

        # evaluate
        transformed_features = [a for a in zip(*feature_dict['valid'])][-1]
        y_pred = None
        if multi_model:
            all_preds = [None] * len(self.models['rf_model'])
            for i in self.models['rf_model']:
                all_preds[i] = self.models['rf_model'][i].predict(
                    transformed_features)
            y_pred = np.array(all_preds).T
        else:
            y_pred = self.models['rf_model'].predict_proba(
                transformed_features)
        right = 0
        wrong = 0
        for i in range(len(y_pred)):
            prediction = np.argmax(y_pred[i])
            if label_dict['valid'][i][prediction] == 1:
                right += 1
            else:
                wrong += 1
        print('Accuracy', right / (right + wrong))
        return

    def preprocessing(self, model_name, data, size=4):
        """
        Transforms features in a format which can be used by the ann model.

        Attributes:
            model_name (str): key of the model in self.models
            data (dict): dictonary of datasets (e.g. train, test and validataion
                data set) where each dataset contains a list of table_ids
                corresponding to the ids in the SQLite database, a list of
                tables (matrix of cells), and a corresponding list of labels
            size (int): number of first rows or columns for which features
                should be obtained
        """

        table_features = defaultdict(list)
        count = 0
        for key in data:
            table_ids = data[key]['table_ids']
            table_data = data[key]['table_data']

            # create feature vectors
            table_features[key] = []
            for id_index, table in enumerate(table_data):
                feature_vector = self. \
                    _get_table_feature_vector_for_lstm(model_name,
                                                       table_ids[id_index],
                                                       table, size, masking=True,
                                                       max_sequence_size=self.MAX_SEQUENCE_SIZE)
                table_features[key].append(feature_vector)

                count += 1
                if count % 100 == 0:
                    print('Preprocessing done for', count, 'tables')
            table_features[key] = table_features[key]

        return table_features

    def label_preprocessing(self, data, label_set):
        """
        Transforms labels into one-hot encoding.

        Attributes:
            data (dict): dictonary of datasets (e.g. train, test and validataion
                data set) where each dataset contains a list of table_ids
                corresponding to the ids in the SQLite database, a list of
                tables (matrix of cells), and a corresponding list of labels
            label_set: set of all possible labels
        """
        # label_list = list(label_set)
        # TODO replace this with comment above
        label_list = ['RELATION', 'OTHER', 'ENTITY', 'MATRIX']
        label_encodings = defaultdict(list)
        for key in data:
            labels = data[key]['labels']
            # crate one hot encoded label vectors
            for label in labels:
                vec = np.zeros(len(label_list))
                pos = label_list.index(label)
                vec[pos] = 1
                label_encodings[key].append(vec)
            label_encodings[key] = np.array(label_encodings[key])
        return label_encodings, label_list

    def _extract_feature_vectors(self, model_name, columns):
        """ Extracts feature vectors for a set of columns.

        Features:
            * Content Pattern: A pattern of length n (default: 10) that
            captures the types of the first n characters in every cell

            * Word Embeddings: Represent cells by a word embedding model

            * Web Table Embeddings: Represnt celss by a web table embeddding
              model

        Returns:
            * features
        """
        features = defaultdict(list)
        for i in range(len(columns)):
            col = columns[i]
            col_features = dict()
            for (name, generator) in self.feature_generators[model_name].items():
                col_features = generator.get_features(col)
                features[name].append(col_features)

        return features

    def _get_table_feature_vector_for_lstm(self, model_name, id, table, size,
                                           masking=False,
                                           max_sequence_size=None):
        """
        Creates features for LSTM network using the feature generators to
        generate embedding features.

        Attributes:
            model_name (str): key of the model in self.models
            id (int): table id in SQLite database
            table (list): columns of one table
            size (int): maximal number of rows and columns for which features
                should be extracted
            masking (bool): if True this function creates feature sequences of
                equal size (`max_sequence_size`). If the table is too small
                missing feature vectors are filled with zero vectors.
            max_sequence_size (int): length of feature sequences used if
                `masking` is enabled.
        """
        feature_dict_column_wise = self._extract_feature_vectors(
            model_name, table)
        transposed_table_data = np.transpose(table)
        feature_dict_row_wise = self._extract_feature_vectors(model_name,
                                                              transposed_table_data)
        features_cols = []
        features_rows = []
        for name in feature_dict_column_wise:
            col_features = feature_dict_column_wise[name]
            col_features = [col_features[i] if i < len(col_features) else [np.zeros(
                col_features[0][0].shape)] for i in range(size)]
            row_features = feature_dict_row_wise[name]
            row_features = [row_features[i] if i < len(row_features) else [np.zeros(
                row_features[0][0].shape)] for i in range(size)]
            if masking:
                if max_sequence_size != None:
                    for i in range(len(col_features)):
                        col_features[i] = col_features[i][:max_sequence_size]
                        while len(col_features[i]) < max_sequence_size:
                            col_features[i].append(
                                np.zeros(col_features[i][0].shape[0]))
                    for i in range(len(row_features)):
                        row_features[i] = row_features[i][:max_sequence_size]
                        while len(row_features[i]) < max_sequence_size:
                            row_features[i].append(
                                np.zeros(row_features[i][0].shape[0]))
                else:
                    raise Exception(
                        'Masking requires max_sequence_size != None')
            col_features = np.concatenate(col_features, axis=1)
            row_features = np.concatenate(row_features, axis=1)
            features_cols.append(col_features)
            features_rows.append(row_features)
        f1 = np.concatenate(features_cols, axis=1)
        f1 = np.array([np.concatenate(
            [f1[i], f1[i] * np.mean([x for x in f1 if np.linalg.norm(x) > 0.0001], axis=0)], axis=0) for i in range(len(f1))])
        f2 = np.flip(f1, 0)
        f3 = np.concatenate(features_rows, axis=1)
        f3 = np.array([np.concatenate(
            [f3[i], f3[i] * np.mean([x for x in f3 if np.linalg.norm(x) > 0.0001], axis=0)], axis=0) for i in range(len(f3))])
        f4 = np.flip(f3, 0)
        feature_vector = [f1, f2, f3, f4]
        # get global features (arff features)
        if self.global_features is not None:
            global_feature_vec = self.global_features.get_vector(id)
            feature_vector.append(global_feature_vec)

        return feature_vector
