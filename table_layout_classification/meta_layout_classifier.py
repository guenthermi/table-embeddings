
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class MetaLayoutClassifier:
    def __init__(self, models, multi_models=[]):
        self.models = models
        self.model_names = list(models.keys())
        self.multi_models = multi_models
        self.model_weights = None
        return

    def set_model_weights(self, weights):
        self.model_weights = weights

    def get_prediction(self, model_name, transformed_features, label_index=None):
        model, model_type = self.models[model_name]
        if model_name in self.multi_models:
            model = model[label_index]
        if model_type == 'structured':
            if label_index != None:
                return [[x] for x in model.predict(transformed_features[-1])]
            else:
                return model.predict_proba(transformed_features[-1])
        elif model_type == 'embedding':
            values = model.predict(transformed_features)
            return values
        else:
            raise Exception('Unknown model type ' + model_type)

    def get_all_predictions(self, features, features_key):
        predictions = dict()
        for model_name in self.model_names:
            # (model, model_type) = self.models[model_name]
            transformed_features = [np.array(a) for a in zip(
                *features[model_name][features_key])]
            if model_name in self.multi_models:
                predictions[model_name] = [None] * \
                    len(self.models[model_name][0])
                for i in self.models[model_name][0]:
                    predictions[model_name][i] = self.get_prediction(
                        model_name, transformed_features, label_index=i)
                predictions[model_name] = np.concatenate(
                    predictions[model_name], axis=1)
            else:
                print('Get predicion for', model_name, '...')
                predictions[model_name] = self.get_prediction(
                    model_name, transformed_features)
        return predictions

    def ensemble(self, features, features_key, predictions=None):
        if predictions is None:
            predictions = self.get_all_predictions(features, features_key)

        weight_ensemble_prediction = None

        if self.model_weights is not None:
            for model_name in self.model_names:
                weight = self.model_weights[self.model_names.index(model_name)]
                weight_ensemble_prediction = predictions[model_name] * \
                    weight if weight_ensemble_prediction is None else weight_ensemble_prediction + \
                    predictions[model_name] * weight

        if weight_ensemble_prediction is not None:
            predictions['weight_ensemble_prediction'] = weight_ensemble_prediction

        return predictions
