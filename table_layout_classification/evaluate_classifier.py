import os
import json
import random
import numpy as np
from collections import defaultdict

from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter

from data_loader import DataLoader
from labeled_corpus_interface import LabeledCorpusInterface
from layout_classifier import LayoutClassifier
from pattern_model import PatternModel
from fasttext_model import FastTextModel
from web_table_multi_embedding_model import WebTableMultiEmbeddingModel
from arff_features import ArffFeatures
from meta_layout_classifier import MetaLayoutClassifier

SHUFFLE_DATA = True

RF_MODEL_WEIGHT = 1
WTE_MODEL_WEIGHT = 1


def store_predictions(filename, scores, table_ids, predictions, labels, label_list):
    prediction_results = defaultdict(dict)
    for model_name in predictions:
        for i in range(len(predictions[model_name])):
            label = label_list[np.argmax(labels[i])]
            prediction = label_list[np.argmax(predictions[model_name][i])]
            prediction_results[model_name][table_ids[i]] = {
                'label': label,
                'prediction': prediction
            }
    output_dict = {
        'scores': scores,
        'prediction_results': prediction_results
    }
    f = open(filename, 'a+')
    f.write(json.dumps(output_dict) + os.linesep)
    return


def calculate_evaluation_score(meta_classifier, features, labels):
    predictions = meta_classifier.ensemble(features, 'valid')
    scores = dict()
    for model_name in predictions:
        right = 0
        wrong = 0
        for i in range(len(labels['valid'])):
            prediction = np.argmax(predictions[model_name][i])
            if labels['valid'][i][prediction] == 1:
                right += 1
            else:
                wrong += 1
        score = right / (right + wrong)
        scores[model_name] = score
    return scores, predictions


def create_layout_classifier(fmodel_file, wmodel_files, arff_features):
    # Init embedding models
    pattern_model = PatternModel()
    fasttext_model = FastTextModel(
        fmodel_file) if fmodel_file is not None else None
    webtable_embedding_model = WebTableMultiEmbeddingModel(
        wmodel_files) if wmodel_files is not None else None

    # Init layout classifier
    classifier = LayoutClassifier(
        pattern_model, fasttext_model, webtable_embedding_model, arff_features)
    return classifier


def train_models(classifier, data, label_set, embedding_model_names, multi_models=[], hsize=20):
    label_encodings, label_list = classifier.label_preprocessing(
        data, label_set)
    models = dict()
    features = dict()
    for model_name in embedding_model_names:
        table_features = None
        table_features = classifier.preprocessing(model_name, data)

        # determine feature dimensionality
        input_dim = table_features['train'][0][0].shape[1]
        global_features_input_dim = table_features['train'][0][-1].shape[0]

        # Train random forrest classifier on structured data
        if classifier.models['rf_model'] == None:
            print('TRAIN RF CLASSIFIER')
            if 'rf_model' in multi_models:
                for i in range(len(label_list)):
                    classifier.create_rf_model(label_index=i)
            else:
                classifier.create_rf_model()
            classifier.train_rf_model(
                table_features, label_encodings, multi_model=('rf_model' in multi_models))
            # add model
            models['rf_model'] = (classifier.models['rf_model'], 'structured')
            # add features
            features['rf_model'] = table_features

        # Train LSTM classifier on embedding model
        if model_name in multi_models:
            for i in range(len(label_list)):
                classifier.create_lstm_model(
                    model_name, input_dim, global_features_input_dim, 1, label_index=i)
        else:
            classifier.create_lstm_model(
                model_name, input_dim, global_features_input_dim, len(label_set))

        classifier.train_lstm_model_with_masking(
            model_name, table_features, label_encodings, multi_model=(model_name in multi_models), epochs=hsize)
        # add model
        models[model_name] = (classifier.models[model_name], 'embedding')
        # add features
        features[model_name] = table_features

    return models, features, label_encodings, label_list


def create_arg_parser():
    parser = ArgumentParser("evaluate_classifier",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve',
                            description='''Evaluates the layout classifier.''')
    parser.add_argument('-i', '--input',
                        help="path to sqlite file of the web table corpus", required=True, nargs=1)
    parser.add_argument('-o', '--output',
                        help="path for output txt file", required=True, nargs=1)
    parser.add_argument('-a', '--arff',
                        help="path arff file with global features", required=True, nargs=1)
    parser.add_argument('-f', '--fmodel',
                        help="path to file of pre-trained fastText model", required=False, nargs=1)
    parser.add_argument('-w', '--wmodel',
                        help="path to file of pre-trained web table embedding model", required=False,
                        nargs='*')
    parser.add_argument('-m', '--model-names',
                        help="list of model names", required=False, nargs='*')
    parser.add_argument('-h', '--hsize',
                        help="hidden layer size", required=False, default=[64], type=int, nargs='*')
    parser.add_argument('-s', '--multi-models', nargs='*', default=[],
                        help="train separate classifier for each class")

    return parser


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()

    # Interface for SQLite database with table data
    corpus_interface = LabeledCorpusInterface(args.input[0])
    data_loader = DataLoader(corpus_interface)

    # Load structural ARFF features
    arff_features = ArffFeatures(args.arff[0])

    data = data_loader.split_data(
        splitting={'train': 0.4, 'test': 0.1, 'valid': 0.5}, shuffle=SHUFFLE_DATA)[0]
    label_set = data_loader.get_label_set()

    # Create layout classifier object that manage layout classification models
    fmodel = args.fmodel[0] if args.fmodel is not None else None
    wmodel = args.wmodel if args.wmodel is not None else None

    classifier = create_layout_classifier(fmodel, wmodel, arff_features)
    all_embedding_model_names = [
        (classifier.models['pattern_model'], 'pattern_model'),
        (classifier.models['web_table_model'], 'web_table_model'),
        (classifier.models['word_embedding_model'], 'word_embedding_model')]

    embedding_model_names = args.model_names if args.model_names is not None else [
        m[1] for m in all_embedding_model_names if m[0] is not None]

    # Train layout classifiers
    models, features, label_encodings, label_list = train_models(
        classifier, data, label_set, embedding_model_names, multi_models=args.multi_models, hsize=args.hsize[0])

    # Create meta layout classifier (voting classifier)
    meta_classifier = MetaLayoutClassifier(
        models, multi_models=args.multi_models)
    model_weights = [RF_MODEL_WEIGHT if (name == 'rf_model') else (
        WTE_MODEL_WEIGHT if name in embedding_model_names else 0) for name in meta_classifier.model_names]
    meta_classifier.set_model_weights(model_weights)

    # Evaluate models
    scores, predictions = calculate_evaluation_score(
        meta_classifier, features, label_encodings)
    print('Scores:', scores)

    # output results to file
    store_predictions(args.output[0], scores, data['valid']['table_ids'],
                      predictions, label_encodings['valid'], label_list)


if __name__ == "__main__":
    main()
