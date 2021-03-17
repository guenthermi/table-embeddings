
import sys
import time
from copy import deepcopy

import mxnet as mx
from mxnet import gluon

import gcn_mxnet as gcn
from feature_generator import FeatureGenerator

HIDDEN_LAYERS = [200, 100]
DIRECTIONS = 8
ACTIVATION_FUNCTION = mx.nd.sigmoid
LEARNING_RATE = 0.001


class DECOClassifier:
    """GNN-based classifier that uses embeddings for classifying the cells in
    spreadsheets of the DECO dataset.
    """

    def __init__(self, dgl_graph, features, num_labels, class_weights):
        self.dgl_graph = dgl_graph
        self.features = features
        self.num_labels = num_labels
        self.class_weights = class_weights
        self._create_gcn_net()
        return

    def train(self, train_ids, train_labels, valid_ids, valid_labels,
              max_epochs=150):
        """Trains the GNN network previously created with the _create_gcn_net()
        function.
        """
        epoch = 0
        loss_stagnation = 0
        duration = []
        while (epoch < max_epochs):
            t0 = time.time()
            pred, train_loss = self._perform_training_step(
                train_ids, train_labels)
            valid_loss = self.loss_fcn(
                mx.nd.log_softmax(mx.nd.array(pred[valid_ids])), valid_labels)
            valid_loss = valid_loss.sum() / len(valid_ids)
            mx.nd.waitall()
            duration.append(time.time() - t0)
            valid_acc = gcn.evaluate(pred[valid_ids], valid_labels)
            epoch += 1
            print(("Epoch {:03d} | Train Loss {:.4f} | "
                   + "Valid Loss {:.4f} | Valid Acc {:.4f} | Time(s) {:.4f}").format(
                epoch, train_loss.asscalar(), valid_loss.asscalar(), valid_acc, duration[-1]))

    def evaluate(self, test_ids, labels):
        """Applies the classifier on the test set and returns
        the predition and the accuracy value."""
        pred = self.net(self.dgl_graph)
        acc = gcn.evaluate(pred[test_ids], labels)
        return mx.nd.softmax(pred), acc

    def _create_gcn_net(self):
        """Creates a graph neural network for the classification task.
        """
        self.net = gcn.Net(self.features.shape[1], HIDDEN_LAYERS, self.num_labels,
                           DIRECTIONS, ACTIVATION_FUNCTION, prefix='GCN')
        self.net.initialize()
        self.loss_fcn = gluon.loss.SoftmaxCELoss(from_logits=True)
        self.optimizer = gluon.Trainer(self.net.collect_params(), 'adam', {
            'learning_rate': LEARNING_RATE, 'wd': 0})
        return

    def _perform_training_step(self, train_ids, train_labels):
        """Executes a training set on the provided training data."""
        train_loss, pred = None, None
        with mx.autograd.record():
            pred = self.net(self.dgl_graph)
            train_loss = self.loss_fcn(mx.nd.log_softmax(
                pred[train_ids]) * self.class_weights, train_labels)
            train_loss = train_loss.sum() / len(train_ids)
        train_loss.backward()
        self.optimizer.step(batch_size=1)
        return pred, train_loss
