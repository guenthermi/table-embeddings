
import numpy as np
import mxnet as mx

import gcn_mxnet as gcn

VOTING_TYPE = 'mean' # 'header_0'

class VotingClassifier:
    """Classifier that combines several classifiers for the cells in
    spreadsheets of the DECO dataset. It uses a simple voting mechanism to
    combine the predictions of the classifiers to a combined prediction.
    """

    def __init__(self, classifiers, all_features, feature_preparation, labels):
        self.classifiers = classifiers
        self.all_features = all_features
        self.feature_preparation = feature_preparation
        self.labels = labels
        self.header_id = self.labels.index('Header')

    def evaluate(self, test_ids, labels):
        """Applies the classifiers on the test set, appies the voting mechanism,
        and returns the combined predition and the accuracy value."""
        # collect predictions
        predictions = []
        for i in range(len(self.classifiers)):
            single_classifier = self.classifiers[i]
            features = self.all_features[i]
            prediction, acc = single_classifier.evaluate(test_ids, labels)
            predictions.append(prediction.asnumpy())
        # combine predicition
        if VOTING_TYPE == 'mean':
            pred = np.mean(predictions, axis=0)
        if VOTING_TYPE == 'header_0':
            pred = np.array([[predictions[0][i][j] if (j == self.header_id) else predictions[1][i][j] for j in range(len(self.labels))] for i in range(len(predictions[0]))])
        pred = mx.nd.array(pred)
        acc = gcn.evaluate(pred[test_ids], labels)
        return pred, acc
