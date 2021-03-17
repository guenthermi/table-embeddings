
import numpy as np
import mxnet as mx
from collections import Counter
from sklearn.ensemble import RandomForestClassifier


class RFClassifier:
    """Random forrest classifier that uses only the feature vectors
    (deco features and/or embedding vectors) for classifying the cells in
    spreadsheets of the DECO dataset.
    """

    def __init__(self, features, num_labels):
        self.features = features.numpy()
        self.num_labels = num_labels
        self.rf_classifier = self._create_rf_classifier()
        return

    def train(self, train_ids, train_labels, valid_ids, valid_labels):
        """Trains the classifier previously created with the _create_rf_classifier()
        function.
        """
        # transform features and labels
        features = self.features[np.array(train_ids.asnumpy(), dtype=int)]
        labels = np.array(train_labels.asnumpy(), dtype=int)
        # train classifier
        self.rf_classifier.fit(features, labels)
        return

    def evaluate(self, test_ids, labels):
        """Applies the classifier on the test set and returns
        the predition and the accuracy value."""
        # predict labels
        pred = self.rf_classifier.predict_proba(self.features)
        # determine accuracy
        results = pred[np.array(test_ids.asnumpy(), dtype=int)]
        indices = np.argmax(results, axis=1)
        labels = labels.asnumpy()
        print(Counter(indices))
        correct = sum(indices == labels)
        acc = correct * 1.0 / len(labels)
        return mx.nd.array(pred), acc

    def _create_rf_classifier(self):
        """Creates sklearn rf classifier for the classification task.
        """
        classifier = RandomForestClassifier(
            n_estimators=100, class_weight='balanced')
        return classifier
