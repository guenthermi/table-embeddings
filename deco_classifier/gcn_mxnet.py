import dgl.function as fn
import mxnet as mx
import mxnet.gluon as gluon
import numpy as np
from collections import Counter

gcn_sum_reduce = fn.sum(msg='m', out='neighbors')


def msg_func(edges):
    f = (edges.src['h'].T)
    a = [
        ((edges.data['direction_features'] == 0) * f).T,
        ((edges.data['direction_features'] == 1) * f).T,
        ((edges.data['direction_features'] == 2) * f).T,
        ((edges.data['direction_features'] == 3) * f).T,
        ((edges.data['direction_features'] == 4) * f).T,
        ((edges.data['direction_features'] == 5) * f).T,
        ((edges.data['direction_features'] == 6) * f).T,
        ((edges.data['direction_features'] == 7) * f).T,

        mx.nd.array(np.atleast_2d(
            ((edges.data['direction_features'] == 0) * (edges.data['distance_features'] < 2)).asnumpy())).T,
        mx.nd.array(np.atleast_2d(
            ((edges.data['direction_features'] == 1) * (edges.data['distance_features'] < 2)).asnumpy())).T,
        mx.nd.array(np.atleast_2d(
            ((edges.data['direction_features'] == 2) * (edges.data['distance_features'] < 2)).asnumpy())).T,
        mx.nd.array(np.atleast_2d(
            ((edges.data['direction_features'] == 3) * (edges.data['distance_features'] < 2)).asnumpy())).T,
        mx.nd.array(np.atleast_2d(
            ((edges.data['direction_features'] == 4) * (edges.data['distance_features'] < 2)).asnumpy())).T,
        mx.nd.array(np.atleast_2d(
            ((edges.data['direction_features'] == 5) * (edges.data['distance_features'] < 2)).asnumpy())).T,
        mx.nd.array(np.atleast_2d(
            ((edges.data['direction_features'] == 6) * (edges.data['distance_features'] < 2)).asnumpy())).T,
        mx.nd.array(np.atleast_2d(
            ((edges.data['direction_features'] == 7) * (edges.data['distance_features'] < 2)).asnumpy())).T,

        mx.nd.array(np.atleast_2d(
            ((edges.data['direction_features'] == 0) * (edges.data['distance_features'] < 3)).asnumpy())).T,
        mx.nd.array(np.atleast_2d(
            ((edges.data['direction_features'] == 1) * (edges.data['distance_features'] < 3)).asnumpy())).T,
        mx.nd.array(np.atleast_2d(
            ((edges.data['direction_features'] == 2) * (edges.data['distance_features'] < 3)).asnumpy())).T,
        mx.nd.array(np.atleast_2d(
            ((edges.data['direction_features'] == 3) * (edges.data['distance_features'] < 3)).asnumpy())).T,
        mx.nd.array(np.atleast_2d(
            ((edges.data['direction_features'] == 4) * (edges.data['distance_features'] < 3)).asnumpy())).T,
        mx.nd.array(np.atleast_2d(
            ((edges.data['direction_features'] == 5) * (edges.data['distance_features'] < 3)).asnumpy())).T,
        mx.nd.array(np.atleast_2d(
            ((edges.data['direction_features'] == 6) * (edges.data['distance_features'] < 3)).asnumpy())).T,
        mx.nd.array(np.atleast_2d(
            ((edges.data['direction_features'] == 7) * (edges.data['distance_features'] < 3)).asnumpy())).T,

        mx.nd.array(np.atleast_2d(
            ((edges.data['direction_features'] == 0) * (edges.data['distance_features'] < 4)).asnumpy())).T,
        mx.nd.array(np.atleast_2d(
            ((edges.data['direction_features'] == 1) * (edges.data['distance_features'] < 4)).asnumpy())).T,
        mx.nd.array(np.atleast_2d(
            ((edges.data['direction_features'] == 2) * (edges.data['distance_features'] < 4)).asnumpy())).T,
        mx.nd.array(np.atleast_2d(
            ((edges.data['direction_features'] == 3) * (edges.data['distance_features'] < 4)).asnumpy())).T,
        mx.nd.array(np.atleast_2d(
            ((edges.data['direction_features'] == 4) * (edges.data['distance_features'] < 4)).asnumpy())).T,
        mx.nd.array(np.atleast_2d(
            ((edges.data['direction_features'] == 5) * (edges.data['distance_features'] < 4)).asnumpy())).T,
        mx.nd.array(np.atleast_2d(
            ((edges.data['direction_features'] == 6) * (edges.data['distance_features'] < 4)).asnumpy())).T,
        mx.nd.array(np.atleast_2d(
            ((edges.data['direction_features'] == 7) * (edges.data['distance_features'] < 4)).asnumpy())).T

    ]
    combined = mx.nd.concat(*a, dim=1)
    return {'m': combined}

class GCNLayer(gluon.Block):
    def __init__(self, in_feats, out_feats, activation):
        super(GCNLayer, self).__init__()
        self.dense = gluon.nn.Dense(out_feats, in_units=in_feats)
        self.activation = activation

    def forward(self, node):
        h = self.dense(node.data['input'])
        if self.activation:
            h = self.activation(h)
        return {'activation': h}

    def set_old_features(self, old_features):
        self.old_features = old_features


class GCN(gluon.Block):
    def __init__(self, in_feats, out_feats, activation, feature_dimension):
        super(GCN, self).__init__()
        self.apply_mod = GCNLayer(
            in_feats + feature_dimension, out_feats, activation)
        self.in_feats = in_feats

    def forward(self, g, features):
        g.ndata['h'] = features
        g.update_all(msg_func, gcn_sum_reduce)
        a = [
            g.ndata['h'],
            g.ndata['neighbors']
        ]
        g.ndata['input'] = mx.nd.concat(*a, dim=1)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('activation')


class Net(gluon.Block):
    def __init__(self, in_feats, num_hidden, num_cat, num_directions, activation, **kwargs):
        super(Net, self).__init__(**kwargs)
        self.num_hidden = num_hidden
        with self.name_scope():
            self.layers = gluon.nn.Sequential()
            self.layers.add(GCN(
                in_feats, num_hidden[0], activation, 24 + num_directions * in_feats))
            for i in range(1, len(num_hidden)):
                self.layers.add(GCN(
                    num_hidden[i - 1], num_hidden[i], activation, 24 + num_directions * num_hidden[i - 1]))
            self.layers.add(
                GCN(num_hidden[-1], num_cat, None, 24 + num_directions * num_hidden[-1]))

    def forward(self, g):
        x = self.layers[0](g, g.ndata['features'])
        for i in np.arange(1, len(self.layers)):
            x = self.layers[i](g, x)
        return x


def evaluate(pred, labels):
    indices = mx.nd.argmax(pred, axis=1)
    print(Counter(indices.asnumpy()))
    correct = mx.nd.sum(indices == labels)
    return (correct * 1.0 / len(labels)).asscalar()
