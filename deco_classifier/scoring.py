from collections import Counter
import mxnet as mx
import numpy as np

class Scoring:
    """Calculates evaluation scores based on the prediction of classifiers for
    the DECO dataset.
    """

    def __init__(self, test_ids, test_labels, pred, labels):
        self.test_ids = np.array(test_ids.asnumpy(),dtype='float32')
        self.test_labels = np.array(test_labels.asnumpy(),dtype='float32')
        self.pred = np.array(pred.asnumpy(),dtype='float32')
        self.labels = np.array(labels)
        self.indices = np.array(mx.nd.argmax(pred[test_ids], axis=1).asnumpy(),dtype='float32')
        print(Counter(self.indices))
        return

    def get_scores(self):
        """Calculates scores for various metrics on the prediction.
        """
        scores = dict()
        scores['Accuracy'] = self._get_accuracy()
        for i, label in enumerate(self.labels):
            scores['Precision_' + label] = self._get_precision(i)
            scores['Recall_' + label] = self._get_recall(i)
        return scores

    def _get_precision(self, label_id):
        is_label = self.indices == label_id
        is_true = self.indices == self.test_labels
        tp = sum(is_label & is_true)
        fp = (sum(self.indices == label_id) - tp)
        return tp / (tp + fp) if (tp + fp) > 0 else 0

    def _get_recall(self, label_id):
        is_label = self.indices == label_id
        is_true = self.indices == self.test_labels
        tp = sum(is_label & is_true)
        fn = sum(self.test_labels == label_id) - tp
        return tp / (tp + fn)

    def _get_accuracy(self):
        correct = np.sum(self.indices == self.test_labels)
        return float(correct * 1.0 / len(self.test_labels))
