import json
import numpy as np
from scipy.stats import hmean
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter

def create_arg_parser():
    parser = ArgumentParser("create_diagram",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve',
                            description='''Evaluates embedding model on unionablity task.''')

    parser.add_argument('-i', '--input',
                        help="path to file with evaluation results", required=True, nargs='*')
    parser.add_argument('-l', '--labels',
                        help="labels of methods for input files", required=True, nargs='*')

    parser.add_argument('-o', '--output',
                        help="path for output image", required=True, nargs=1)

    parser.add_argument('-h', '--headers',
                        help="take header similarity into account", nargs='?', const=True, default=False)

    return parser


def create_pr_curve_diagram(precision_values, recall_values, labels,
                            output_filename, interpolation=True):
    plt.figure(figsize=(4.5, 5.5))
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    for recall, precision, label in zip(recall_values, precision_values, labels):
        if interpolation:
            pr_values = list(zip(recall, precision))
            pr_values.sort()
            recall, precision = zip(*pr_values)
            precision = [max(precision[i:]) for i in range(len(precision))]
        auc_value = auc(recall, precision)
        print(label, ':', auc_value)
        plt.plot(recall, precision, label=label + ' (' + str(round(auc_value, 3)) + '*)')
    ax = plt.gca()
    handles, _ = ax.get_legend_handles_labels()
    auc_note = mpatches.Patch(color='white', alpha=0.0, label='* AUC')
    handles.append(auc_note)
    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    plt.xlim(0, 1)
    plt.grid(linestyle='--')
    plt.legend(ncol=1, labelspacing=0.05, columnspacing=0.05, fontsize=14, handles=handles)
    plt.subplots_adjust(left=0.18, right=0.96, top=0.990, bottom=0.12)
    plt.savefig(output_filename)
    return


def calculate_pr_curve(p_values, n_values, score_function):
    precision = []
    recall = []

    if score_function == 'data':
        all_scores = [(x[0], 1) for x in p_values] + [(x[0], 0) for x in n_values]
    elif score_function == 'header':
        all_scores = [(x[1], 1) for x in p_values] + [(x[1], 0) for x in n_values]
    else:
        print('Unknown score function:', score_function)
    all_scores.sort(reverse=True)
    tp = 0
    fp = 0
    for i, (score, label) in enumerate(all_scores):
        if label == 1:
            tp += 1
        else:
            fp += 1
        tn = len(n_values) - fp
        fn = len(p_values) - tp
        precision.append((tp / (tp + fp) if tp > 0 else 0))
        recall.append((tp / (tp + fn) if tp > 0 else 0))
    return precision, recall


def load_data(input_path):
    f = open(input_path)
    data = json.load(f)
    return data


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()

    if len(args.input) != len(args.labels):
        print('ERROR different number of inputs and labels')
        quit()

    precision_values = list()
    recall_values = list()
    for input in args.input:
        eval_data = load_data(input)

        p_values = [x[2] for x in eval_data['p_samples']]
        n_values = [x[2] for x in eval_data['n_samples']]

        score_function = 'header' if args.headers else 'data'

        precision, recall = calculate_pr_curve(p_values, n_values, score_function)
        precision_values.append(precision)
        recall_values.append(recall)

    create_pr_curve_diagram(
        precision_values, recall_values, args.labels, args.output[0])

    return


if __name__ == "__main__":
    main()
