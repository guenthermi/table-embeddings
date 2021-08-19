import numpy as np
import json
import random
from turl_embedding_model import TurlEmbeddingModel
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from dataset_loader import DatasetLoader

from web_table_embedding_model import WebTableEmbeddingModel
from fasttext_embedding_model import FasttextEmbeddingModel
from tapas_embedding_model import TapasEmbeddingModel


def create_arg_parser():
    parser = ArgumentParser("calculate_unionablity",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve',
                            description='''Evaluates embedding model on unionablity task.''')

    parser.add_argument('-e', '--embedding-model',
                        help="path to embedding model", required=True, nargs=1)
    parser.add_argument('-et', '--embedding-type',
                        help="embedding type: 'web-table', 'fasttext', or 'word2vec'", required=True, nargs=1)
    parser.add_argument('-o', '--output',
                        help="path for output txt file", required=True, nargs=1)
    parser.add_argument('-b', '--benchmark',
                        help="path to unionablity benchmark folder", required=True, nargs=1)
    parser.add_argument('-s', '--sample-size',
                        help="number of evaluation samples", required=True, nargs=1)
    parser.add_argument('-h', '--model-headers',
                        help="calculate vectors for header terms", nargs='?', const=True, default=False)
    parser.add_argument('-n', '--negative-sample-factor',
                        help="factor that determine number of negative samples in comparison to positive samples", nargs=1, default=[2])

    return parser


def load_embedding_model(model_type, model_path):
    model = None
    if model_type == 'web-table':
        model = WebTableEmbeddingModel(model_path)
    elif model_type == 'fasttext':
        model = FasttextEmbeddingModel(model_path)
    elif model_type == 'tapas':
        model = TapasEmbeddingModel(model_path)
    elif model_type == 'turl':
        model = TurlEmbeddingModel()
    return model


def create_samples(dataset, sample_size=100, n_sample_rate=2):
    alignments, alignments_reverse = dataset.get_alignments()
    query_columns = list(alignments.keys())
    all_columns = list(alignments_reverse.keys())
    p_samples = list()
    n_samples = list()
    while len(p_samples) < sample_size:
        query_table_name, query_col_name = random.choice(query_columns)
        text_values_q = dataset.get_column(query_table_name, query_col_name)
        text_values_p = None
        text_values_n = None
        candidate_list = list([x for x in alignments[(
            query_table_name, query_col_name)] if x[0] != query_table_name])
        if len(candidate_list) == 0:
            continue
        pos_candidate = random.choice(candidate_list)
        try:
            text_values_p = dataset.get_column(
                pos_candidate[0], pos_candidate[1])
        except:
            continue
        p_samples.append(
            (query_col_name, pos_candidate[1], text_values_q, text_values_p))

        for i in range(n_sample_rate):
            text_values_n = None
            while text_values_n is None:
                neg_candidate = random.choice(all_columns)
                if neg_candidate in alignments[(query_table_name, query_col_name)]:
                    continue
                try:
                    text_values_n = dataset.get_column(
                        neg_candidate[0], neg_candidate[1])
                except:
                    continue

            n_samples.append(
                (query_col_name, neg_candidate[1], text_values_q, text_values_n))
    return p_samples, n_samples


def evaluate(model, p_samples, n_samples, model_headers=False):
    results = dict()
    for sample_set, label in [(p_samples, 'p_samples'), (n_samples, 'n_samples')]:
        results[label] = list()
        for (col_name_q, col_name_c, text_values_q, text_values_c) in sample_set:
            score = model.get_approximated_unionability_score(
                text_values_q, text_values_c, col_name_q, col_name_c, model_headers=model_headers)
            results[label].append((col_name_q, col_name_c, score))
    return results


def output_results(results, output_path):
    output_file = open(output_path, 'w')
    json.dump(results, output_file)
    return


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()
    dataset = DatasetLoader(args.benchmark[0])
    model = load_embedding_model(
        args.embedding_type[0], args.embedding_model[0])
    p_samples, n_samples = create_samples(
        dataset, sample_size=int(args.sample_size[0]), n_sample_rate=int(args.negative_sample_factor[0]))
    results = evaluate(model, p_samples, n_samples,
                       model_headers=args.model_headers)
    output_results(results, args.output[0])

    return


if __name__ == "__main__":
    main()
