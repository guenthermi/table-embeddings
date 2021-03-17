import numpy as np
import json

from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter

from taxonomy import Taxonomy
from fasttext_embedding_model import FasttextEmbeddingModel
from web_table_embedding_model import WebTableEmbeddingModel
from word2vec_model import Word2VecModel

EPSILON = 1e-10  # only to prevent division by zero


def create_arg_parser():
    parser = ArgumentParser("evaluate_model",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve',
                            description='''Evaluates embedding model on yago taxonomy.''')

    parser.add_argument('-e', '--embedding-model',
                        help="path to fastText embedding model", required=False, nargs=1)
    parser.add_argument('-o', '--output',
                        help="path for output txt file", required=False, nargs=1)
    parser.add_argument('-t', '--taxonomy',
                        help="path to taxonomy folder", required=True, nargs=1)
    parser.add_argument('-et', '--embedding-type',
                        help="embedding type: 'web-table', 'fasttext', or 'word2vec'", required=True, nargs=1)
    parser.add_argument('-s', '--sample-size',
                        help="number of evaluation samples", required=False, nargs=1)
    parser.add_argument('-l', '--load-pickle',
                        help="load taxonomy from pickle file", nargs='?', const=True, default=False)
    parser.add_argument('-n', '--negative-sample-factor',
                        help="factor that determine number of negative samples in comparison to positive samples", nargs=1, default=[1])

    return parser


def calculate_pr_curve(sim_values, labels):
    precision = []
    recall = []
    labels = np.array(labels, dtype='int')
    for th in np.arange(-1, 1, 0.01):
        pred = np.array(sim_values >= th, dtype='int')
        recall.append((np.sum(pred & labels) + EPSILON) /
                      (np.sum(labels) + EPSILON))
        precision.append((np.sum(pred & labels) + EPSILON) /
                         (np.sum(pred) + EPSILON))
    return precision, recall


def evaluate_model(inst_vecs, class_vecs, labels, neg_factor):
    # calculate score
    sim_values = np.array([np.dot(inst_vecs[i], class_vecs[i])
                           for i in range(labels.shape[0])])
    precision, recall = calculate_pr_curve(sim_values, labels)
    return precision, recall


def export_eval_results(precision, recall, filename):
    f = open(filename, 'w')
    eval_data = {
        "precision": precision,
        "recall": recall
    }
    json.dump(eval_data, f)
    f.close()
    return


def get_vectors(embedding_model, samples):
    instance_values, class_values, labels = zip(*samples)
    instance_embeddings = np.array([embedding_model.get_instance_vector(
        value, norm=True) for value in instance_values])
    class_embeddings = np.array([embedding_model.get_class_vector(
        value, norm=True) for value in class_values])
    labels = np.array(labels)
    return instance_embeddings, class_embeddings, labels


def evaluate(args):
    taxonomy = Taxonomy(args.taxonomy[0])
    if args.load_pickle:
        print('Load taxonomy ...')
        taxonomy.load_taxonomy()
    else:
        taxonomy.construct_taxonomy()
        taxonomy.save_taxonomy()

    print('Create sample set ...')
    samples = taxonomy.sample_links(
        int(args.sample_size[0]), int(int(args.sample_size[0]) *
                                      float(args.negative_sample_factor[0])))

    print('Load embedding model ...')
    embedding_model = None
    if args.embedding_type[0].lower() == 'web-table':
        embedding_model = WebTableEmbeddingModel(args.embedding_model[0])
    elif args.embedding_type[0].lower() == 'fasttext':
        embedding_model = FasttextEmbeddingModel(args.embedding_model[0])
    elif args.embedding_type[0].lower() == 'word2vec':
        embedding_model = Word2VecModel(
            args.embedding_model[0], model_format='word2vec')
    elif args.embedding_type[0].lower() == 'glove':
        embedding_model = Word2VecModel(
            args.embedding_model[0], model_format='glove')
    else:
        print('Unkown embedding model type:', args.embedding_type[0])
        quit()

    print('Create embedding vectors ...')
    instance_embeddings, class_embeddings, labels = get_vectors(
        embedding_model, samples)

    print('Evaluate embedding model ...')
    precision, recall = evaluate_model(
        instance_embeddings, class_embeddings, labels,
        float(args.negative_sample_factor[0]))

    print('Export embedding results ...')
    export_eval_results(precision, recall, args.output[0])
    return


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()
    evaluate(args)
    return


if __name__ == "__main__":
    main()
