
import sys
import random
from collections import Counter

import numpy as np
from mxnet import nd
import networkx as nx

import torch
import torch.nn.functional as F

import dgl

FEATURE_SCALING = False


class FeaturePreparation:
    def __init__(self, feature_generator, embedding_model):
        self.embedding_model = embedding_model
        self.vector_lookup = self.get_embedding_lookup(
            feature_generator.features)
        self.raw_feature_data = feature_generator.features
        self.graphs = feature_generator.graphs
        self.label_data = feature_generator.labels
        self.node_attributes = feature_generator.node_attributes
        return

    def sample_sheets(self, splitting):
        """Samples set of sheets with a size according to the relative size
        values in the splitting parameter.
        Returns an array of arrays with sheet keys and a dictonary which
        maps those sheet keys to the sheet graphs.
        """
        sheet_keys = list(self.graphs.keys())
        random.shuffle(sheet_keys)
        size = len(sheet_keys)
        print('Number of Sheets:', size)
        splitted_sheet_keys = []
        bound = 0
        for split in splitting:
            next_bound = bound + int(split * size)
            splitted_sheet_keys.append(sheet_keys[bound:next_bound])
            bound = next_bound
        graphs = []
        for key_list in splitted_sheet_keys:
            for key in key_list:
                graphs.append(self.graphs[key])
        return splitted_sheet_keys, graphs

    def construct_dgl_graph(self, graphs):
        print('Create combined networkx graph ...')
        combined_graph = nx.DiGraph()
        for i, graph in enumerate(graphs):
            relabel = {n: (str(i) + '_' + n) for n in graph.nodes()}
            ng = nx.relabel_nodes(graph, relabel)
            combined_graph.add_nodes_from(ng.nodes(data=True))
            combined_graph.add_edges_from(ng.edges(data=True))
            print('\rAdd Graphs %d/%d' % (i + 1, len(graphs)), end='')
        print()
        print('Convert nx to dgl graph ...')
        g = dgl.DGLGraph()
        g = dgl.from_networkx(combined_graph, node_attrs=[
                        'node_id'], edge_attrs=['direction', 'distance'])
        return g

    def construct_features_for_dgl_graph(self, dgl_graph, feature_type):
        feature_vectors = []
        deco_dims = 0
        missing_nodes = 0
        for dgl_node_id, node_id in enumerate(dgl_graph.ndata['node_id'].asnumpy()):
            attrs = self.node_attributes[node_id]
            sheet_key = (attrs['filename'], attrs['sheetname'])
            cell_key = (attrs['col'], attrs['row'])
            text_value = self.raw_feature_data[sheet_key][cell_key]['content']
            col_vec = np.mean([self.vector_lookup[t] for t in self.raw_feature_data[sheet_key][cell_key]['col_values']],axis=0)
            row_vec = np.mean([self.vector_lookup[t] for t in self.raw_feature_data[sheet_key][cell_key]['row_values']],axis=0)
            embedding_vector = np.concatenate([self.vector_lookup[text_value], col_vec, row_vec])
            if 'deco_features' in self.raw_feature_data[sheet_key][cell_key]:
                deco_features = self.raw_feature_data[sheet_key][cell_key]['deco_features']
                deco_dims = len(deco_features)
                if np.linalg.norm(deco_features) == 0:
                    missing_nodes += 1
                    print('WARNING: Missing feature for',
                          cell_key, 'in', sheet_key, '(' + text_value + ')')
                if feature_type == 'deco':
                    feature_vectors.append(deco_features)
                elif feature_type == 'embeddings':
                    feature_vectors.append(embedding_vector)
                elif feature_type == 'combined':
                    vector = np.concatenate(
                        [deco_features, embedding_vector])
                    feature_vectors.append(vector)
                else:
                    print('ERROR: Unkown feature_type:', feature_type)
                    quit()
            else:
                if feature_type == 'deco':
                    print('ERROR: DECO features are missing.')
                feature_vectors.append(embedding_vector)
        print('Missing Nodes:', missing_nodes,
              'Missing Rate:', missing_nodes / int(dgl_graph.ndata['node_id'].shape[0]))
        # feature selection and feature scaling
        feature_vectors = self._feature_selection_and_scaling(
            np.array(feature_vectors), scaling_dims=range(deco_dims))
        print('Extracted %d features' % (feature_vectors[0].shape[0],))
        return torch.FloatTensor(feature_vectors)

    def get_embedding_lookup(self, features):
        result = dict()
        for sheet_key, sheet_features in features.items():
            for cell_key, feature_dict in sheet_features.items():
                text_value = feature_dict['content']
                if text_value not in result:
                    result[text_value] = self.embedding_model.get_features(
                        text_value)
        return result

    def add_features_to_graph(self, dgl_graph, features, normalization=False):
        """Adds features to the graph and normalize them if applicabe"""
        features = F.normalize(
            features, p=2, dim=1) if normalization else features
        features = nd.array(features)
        dgl_graph.ndata['features'] = features
        dgl_graph.edata['direction_features'] = nd.array(
            dgl_graph.edata['direction'], dtype='float32')
        dgl_graph.edata['distance_features'] = nd.array(
            dgl_graph.edata['distance'], dtype='float32')
        return dgl_graph

    def create_node_label_lookup(self, dgl_graph):
        """Returns node_id->label_id dictonary and a list with all labels where
        positons correspond to label_ids
        """
        labels = dict()
        for node_id in dgl_graph.ndata['node_id'].asnumpy():
            # node_id = node['node_id']
            attrs = self.node_attributes[node_id]
            labels[node_id] = self.label_data[(
                attrs['filename'], attrs['sheetname'])][attrs['col'], attrs['row']]
        all_labels = list(set(labels.values()))
        id_lookup = dict([(l, i) for i, l in enumerate(all_labels)])
        label_lookup = dict()
        for node_id, label in labels.items():
            label_lookup[node_id] = id_lookup[label]
        return label_lookup, all_labels

    def create_mx_arrays(self, dgl_graph, label_lookup, train_sheets, valid_sheets, test_sheets, downsampling=False):
        """Gets dgl_graph_node_ids for node_ids in train_sheets, valid_sheets,
        and test_sheets and construct vectors for ids and labels.
        """

        train_ids = []
        train_labels = []
        valid_ids = []
        valid_labels = []
        test_ids = []
        test_labels = []
        for dgl_node_id, node_id in enumerate(dgl_graph.ndata['node_id'].asnumpy()):
            attributes = self.node_attributes[node_id]
            if (attributes['filename'], attributes['sheetname']) in train_sheets:
                train_ids.append(dgl_node_id)
                train_labels.append(label_lookup[node_id])
            if (attributes['filename'], attributes['sheetname']) in valid_sheets:
                valid_ids.append(dgl_node_id)
                valid_labels.append(label_lookup[node_id])
            if (attributes['filename'], attributes['sheetname']) in test_sheets:
                test_ids.append(dgl_node_id)
                test_labels.append(label_lookup[node_id])

        if downsampling:
            train_ids, train_labels = self._downsampling(
                train_ids, train_labels, strategy='soft')

        weight_vector = self._get_class_weights(train_labels, method='equal')

        train_ids = nd.array(train_ids).astype(np.int64)
        valid_ids = nd.array(valid_ids).astype(np.int64)
        test_ids = nd.array(test_ids).astype(np.int64)

        train_labels = nd.array(train_labels)
        valid_labels = nd.array(valid_labels)
        test_labels = nd.array(test_labels)

        return train_ids, train_labels, valid_ids, valid_labels, test_ids, test_labels, weight_vector

    def _downsampling(self, train_ids, train_labels, strategy='soft'):
        # get label positions
        label_positions = dict()
        for s in set(train_labels):
            label_positions[s] = [
                i for (i, l) in enumerate(train_labels) if l == s]

        # determine cardinalities
        min_cardinality = min([len(label_positions[s])
                               for s in label_positions])
        avg_cardinality = len(train_labels) / len(label_positions.keys())
        cardinality = None
        if strategy == 'soft':
            cardinality = int(avg_cardinality)
        elif strategy == 'hard':
            cardinality = int(min_cardinality)
        else:
            print('ERROR: Unkown downsampling strategy:', strategy)
            quit()

        # sample
        for key, positions in label_positions.items():
            random.shuffle(positions)

        label_positions = {k: v[:cardinality]
                           for (k, v) in label_positions.items()}

        # determine downsampled train_ids and train_labels
        new_labels = [(pos, l)
                      for l in label_positions for pos in label_positions[l]]
        new_labels.sort()
        new_ids = [train_ids[pos] for (pos, l) in new_labels]
        new_labels = [l for (pos, l) in new_labels]
        return new_ids, new_labels

    def _get_class_weights(self, train_labels, method='equal'):
        c = Counter(train_labels)
        weight_vector = np.zeros(len(c))
        if method == 'log':
            for label in c:
                weight_vector[label] = np.log(1.0 / (c[label] / len(train_labels)))
        elif method == 'equal':
            weight_vector = np.ones(len(c))
        else:
            print('ERROR: Unkown weighting method:', method)
            quit()
        return nd.array(weight_vector)

    def _feature_selection_and_scaling(self, feature_vectors, scaling_dims=None):
        selection = []
        for i, col in enumerate(feature_vectors.T):
            if len(np.unique(col)) > 1:
                selection.append(i)
        # feature selection
        # z scoring
        if scaling_dims is None:
            scaling_dims = range(len(feature_vectors[0]))
        scaling_dims = set(scaling_dims).intersection(selection)
        if FEATURE_SCALING:
            offsets = np.zeros(len(feature_vectors[0]))
            scale_factors = np.ones(len(feature_vectors[0]))

            constant_dims = set()
            for i, col in enumerate(feature_vectors.T):
                if i in scaling_dims:
                    offsets[i] = np.mean(col)
                    std = np.std(col)
                    if std == 0:
                        constant_dims.add(i)
                    elif std < 1.0:
                        scale_factors[i] = 1.0
                        offsets[i] = 0
                    else:
                        scale_factors[i] = max(std, 1)

            new_selection = []
            for elem in selection:
                if elem not in constant_dims:
                    new_selection.append(elem)
                else:
                    scale_factors[elem] = 1
            return ((feature_vectors - offsets) / scale_factors)[:, new_selection]
        else:
            return feature_vectors[:, selection]
