
import numpy as np
import mxnet as mx

from classifier import DECOClassifier
import gcn_mxnet as gcn


class DECOClassifierOVA:
    """GNN-based classifier that uses embeddings for classifying the cells in
    spreadsheets of the DECO dataset. It is based on several binary classifiers
    combined by a one-vs-all strategy to solve this
    mulit-class classification problem.
    """

    def __init__(self, dgl_graph, features, num_labels):
        self.gnn_classifiers = self._create_classifiers(
            dgl_graph, features, num_labels)
        return

    def train(self, train_ids, train_labels, valid_ids, valid_labels,
              max_epochs=300, max_stagnation=50):
        """Trains the GNN networks previously created with the create_gcn_net()
        function.
        """
        for i, classifier in enumerate(self.gnn_classifiers):
            bin_train_labels = self._get_binary_labels(train_labels, i)
            bin_valid_labels = self._get_binary_labels(valid_labels, i)
            classifier.train(
                train_ids, bin_train_labels, valid_ids, bin_valid_labels)
        return

    def evaluate(self, test_ids, labels):
        """Applies the classifier on the test set and returns
        the predition and the accuracy value."""
        results = []
        for i, classifier in enumerate(self.gnn_classifiers):
            pred = mx.nd.softmax(classifier.net(classifier.dgl_graph))
            results.append(pred[:, 1].expand_dims(1))
        results = mx.nd.concat(*results, dim=1)
        acc = gcn.evaluate(results[test_ids], labels)
        return mx.nd.softmax(results), acc

    def _create_classifiers(self, dgl_graph, features, num_labels):
        weight_vector = mx.nd.array(np.ones(2)) # no class weights
        classifiers = list()
        for i in range(num_labels):
            classifiers.append(DECOClassifier(dgl_graph, features, 2, weight_vector))
        return classifiers

    def _get_binary_labels(self, labels, selected_label):
        """Creates a binary label vector from a mulit-class label vector.
        """
        return (labels == selected_label)
