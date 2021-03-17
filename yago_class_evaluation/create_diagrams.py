import json
import matplotlib.pyplot as plt

from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter


def create_arg_parser():
    parser = ArgumentParser("create_diagrams",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve',
                            description='''Plots precision-recall diagrams for evaluation results''')

    parser.add_argument('-i', '--inputs',
                        help="paths to json files with evaluation results", required=False, nargs='*')
    parser.add_argument('-l', '--labels',
                        help="labels of evaluated models", required=False, nargs='*')
    parser.add_argument('-o', '--output',
                        help="path to output folder", required=False, nargs=1)
    parser.add_argument('-ip', '--interpolation',
                        help="calculate the interpolated precision recall curve", nargs='?', const=True, default=False)

    return parser


def parse_inputs(inputs):
    precision_values = []
    recall_values = []
    for input in inputs:
        f = open(input)
        data = json.load(f)
        f.close()
        precision_values.append(data['precision'])
        recall_values.append(data['recall'])
    return precision_values, recall_values


def create_pr_curve_diagram(precision_values, recall_values, labels,
                            output_filename, interpolation=False):
    plt.figure(figsize=(8, 5.5))
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    for recall, precision, label in zip(recall_values, precision_values, labels):
        if interpolation:
            pr_values = list(zip(recall, precision))
            pr_values.sort()
            recall, precision = zip(*pr_values)
            precision = [max(precision[i:]) for i in range(len(precision))]
        plt.plot(recall, precision, label=label)
    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    plt.xlim(0, 1)
    plt.grid(linestyle='--')
    plt.subplots_adjust(left=0.1, right=0.98, top=0.990, bottom=0.12)
    plt.legend(labelspacing=0.05, fontsize=14)
    plt.savefig(output_filename)
    return


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()
    precision_values, recall_values = parse_inputs(
        args.inputs)
    create_pr_curve_diagram(precision_values, recall_values, args.labels,
                            args.output[0], interpolation=args.interpolation)
    return


if __name__ == "__main__":
    main()
