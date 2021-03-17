
import sys
import networkx as nx
import pickle
from collections import defaultdict

from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter

from annotation_parser import AnnotationParser
from sheet_parser import SheetParser
from deco_features_interface import DECOFeaturesInterface

from cell_object import CellObject

DEFAULT_MAX_SHEET_SIZE = 100  # maximal number of cells in the sheet


class FeatureGenerator:
    """Provides features for DECO Classifier.
    This class is able to create features from spreadsheet files and
    annotation files using the dedicated interface classes.
    """

    def __init__(self, annotation_file_path=None, sheet_folder=None,
                 feature_file_path=None, feature_description_file_path=None,
                 pickle_file_path=None, max_sheet_size=DEFAULT_MAX_SHEET_SIZE):
        self.graphs = dict()
        self.features = dict()
        self.labels = dict()
        self.graphs = dict()
        self.node_attributes = dict()
        self.directions = ['l', 'lt', 't', 'rt', 'r', 'rb', 'b', 'lb']
        self.sheets = None
        self.sheet_lookup = None
        self.sheet_set = None
        self.max_sheet_size = max_sheet_size
        if pickle_file_path is not None:
            self.load_from_file(pickle_file_path)
        if annotation_file_path is not None:
            print('Load annotations ...')
            self.annotations = self._load_annotations(annotation_file_path)

            # determine sheets smaller than max_sheet_size
            self.sheet_set = set()
            for filename in self.annotations:
                for sheetname in self.annotations[filename]:
                    if len(self.annotations[filename][sheetname].index) < (self.max_sheet_size):
                        self.sheet_set.add((filename, sheetname))

        if sheet_folder is not None:
            print('Load sheet data ...')
            self.sheet_data = self._load_sheet_data(sheet_folder)
        if (feature_file_path is not None) and (
                feature_description_file_path is not None):
            self.deco_features = DECOFeaturesInterface(
                feature_file_path, feature_description_file_path, sheet_set=self.sheet_set)

        return

    def create_labels(self, simple=False):
        """Constructs a label lookup for every sheet that maps the
        cell position (col, row) the label given in the annotations file.

        Returns a dictonary that maps (filename, sheet_name) keys to those
        lookups.
        """
        simple_label = {
            'Data': 'Data',
            'Derived': 'Data',
            'GroupHead': 'Data',
            'Header': 'Header',
            'MetaTitle': 'MetaData',
            'Notes': 'MetaData',
            'Other': 'MetaData'
        }
        label_func = (lambda x: simple_label[x]) if simple else (lambda x: x)
        self.labels = dict()
        for i, filename in enumerate(self.annotations):
            for sheet_name in self.annotations[filename]:
                df = self.annotations[filename][sheet_name]
                sheet_labels = {
                    (row[0], row[1]): label_func(row[2]) for row in zip(df['FirstColumn'], df['FirstRow'], df['AnnotationLabel'])}
                self.labels[(filename, sheet_name)] = sheet_labels
        return self.labels

    def get_graph(self):
        """Constructs graphs of neighboring nodes for all sheets.
        Each node in the graph hold attributes for column, row, file name, and
        sheet name.

        Returns a dictonary that maps each sheet (filename, sheet_name) to the
        corresponding graph.
        """
        RADIUS = 4
        self.graphs = dict()
        self.node_attributes = dict()
        next_node_id = 0
        get_direction = dict([(x, i) for i, x in enumerate(self.directions)])
        for (filename, sheetname) in self.labels:
            node_names = set()
            G = nx.DiGraph()
            sheet_labels = self.labels[(filename, sheetname)]
            for (col, row) in sheet_labels:
                node_name = self._get_node_name(col, row)
                if node_name not in node_names:
                    node_id = next_node_id
                    next_node_id += 1
                    self.node_attributes[node_id] = {
                        'col': col,
                        'row': row,
                        'filename': filename,
                        'sheetname': sheetname
                    }
                    G.add_node(node_name, node_id=node_id)
                neighbor_candidates = [
                    # col, row, direction
                    (-1, 0, 'l'),
                    (-1, -1, 'lt'),
                    (0, -1, 't'),
                    (1, -1, 'rt'),
                    (1, 0, 'r'),
                    (1, 1, 'rb'),
                    (0, 1, 'b'),
                    (-1, 1, 'lb')]
                for candidate in neighbor_candidates:
                    for n in range(1, RADIUS):
                        target_col = candidate[0] * n + col
                        target_row = candidate[1] * n + row
                        query = (target_col, target_row)
                        if query in sheet_labels:
                            target_name = self._get_node_name(
                                target_col, target_row)
                            if target_name not in node_names:
                                node_id = next_node_id
                                next_node_id += 1
                                self.node_attributes[node_id] = {
                                    'col': target_col,
                                    'row': target_row,
                                    'filename': filename,
                                    'sheetname': sheetname
                                }
                                G.add_node(target_name, node_id=node_id)
                            G.add_edge(node_name, target_name,
                                       direction=get_direction[candidate[2]], distance=n)
                            break
            self.graphs[(filename, sheetname)] = G
        return self.graphs, self.node_attributes, self.directions

    def get_features(self, deco_default_vector='zero'):
        """Creates features for labeled cells in every sheet.
        Returns a dictionary (keys: (filename, sheetname)) that hold itself a
        dictonary for every sheet that maps cells (col, row) to a
        dictonary of features.
        """
        if len(self.labels) == 0:
            print('WARNING: `labels` is empty')
        self.features = dict()
        sheet_set = self.sheet_set if self.sheet_set is not None else self.labels.keys()
        for (filename, sheetname) in sheet_set:
            sheet = self.sheet_data[filename].sheet_by_name(sheetname)
            sheet_features = dict()
            row_values = defaultdict(list)
            col_values = defaultdict(list)
            for (col, row) in self.labels[(filename, sheetname)]:
                cell = sheet.cell(row, col)
                deco_feature_vector = self.deco_features.get_feature_vector(
                    filename, sheetname, col, row, default=deco_default_vector)
                sheet_features[(col, row)] = {
                    'ctype': cell.ctype,
                    'content': str(cell.value),
                    'deco_features': deco_feature_vector
                }
                row_values[row].append(str(cell.value))
                col_values[col].append(str(cell.value))
            sheet_feature_keys = list(sheet_features.keys())
            for col, row in sheet_feature_keys:
                sheet_features[(col, row)]['col_values'] = col_values[col]
                sheet_features[(col, row)]['row_values'] = row_values[row]
            self.features[(filename, sheetname)] = sheet_features
        return self.features

    def remove_empty_cells(self):
        """Remove cells from labels which are empty.
        """
        count = 0
        for sheet_key in self.features:
            sheet_labels = self.labels[sheet_key]
            to_remove = list()
            for cell_key in sheet_labels:
                if cell_key not in self.features[sheet_key]:
                    print('ERROR feature for cell', cell_key, 'in', sheet_key, 'is missing')
                    quit()
                text_value = self.features[sheet_key][cell_key]['content']
                if len(text_value) < 1:
                    count += 1
                    to_remove.append(cell_key)
            # remove cells
            for cell_key in to_remove:
                del self.labels[sheet_key][cell_key]
        print('Removed', count, 'empty cells')
        return

    def remove_no_feature_cells(self):
        """Removes cells from labels not present in the feature file.
        """
        count = 0
        for sheet_key in self.features:
            sheet_labels = self.labels[sheet_key]
            to_remove = list()
            for cell_key in sheet_labels:
                if cell_key not in self.features[sheet_key]:
                    print('ERROR feature for cell', cell_key, 'in', sheet_key, 'is missing')
                    quit()
                vector = self.features[sheet_key][cell_key]['deco_features']
                if vector is None:
                    count += 1
                    to_remove.append(cell_key)
            # remove cells
            for cell_key in to_remove:
                del self.labels[sheet_key][cell_key]
        print('Removed', count, 'cells without a feature')
        return

    def export_feature_data(self, filename):
        """Exports extracted features into a python pickle file
        """
        f = open(filename, 'wb')
        data = {
            'features': self.features,
            'labels': self.labels,
            'graphs': self.graphs,
            'node_attributes': self.node_attributes,
            'directions': self.directions,
            'max_sheet_size': self.max_sheet_size
        }
        pickle.dump(data, f)
        return

    def load_from_file(self, file_path):
        """Imports features from a python pickle file preveously exported with
        the export_feature_data() method.
        """
        ONLY_SMALL_GRAPHS = True
        f = open(file_path, 'rb')
        p_data = pickle.load(f)
        self.max_sheet_size = p_data['max_sheet_size']
        self.features = p_data['features']
        self.labels = p_data['labels']
        self.graphs = dict()
        if ONLY_SMALL_GRAPHS:
            for key, graph in p_data['graphs'].items():
                if len(graph.nodes()) < self.max_sheet_size:
                    self.graphs[key] = graph
                else:
                    if key in self.features:
                        del self.features[key]
                    if key in self.labels:
                        del self.labels[key]
        else:
            self.graphs = p_data['graphs']
        self.node_attributes = p_data['node_attributes']
        self.directions = p_data['directions']
        print(('Loaded feature from pickle file:\n'
               'Features:       %d\n'
               'Lablels:        %d\n'
               'Graphs:         %d\n'
               'Node Attributes %d\n'
               'Directions:     %d'
               ) % (len(self.features), len(self.labels), len(self.graphs),
                    len(self.node_attributes), len(self.directions)))
        self.get_label_distribution()
        return

    def get_label_distribution(self):
        """Determines the distribution of labels in the dataset and print it
        to the command line.
        """
        distribution = defaultdict(int)
        for sheet_key in self.labels:
            for cell_key in self.labels[sheet_key]:
                label = self.labels[sheet_key][cell_key]
                distribution[label] += 1
        n = sum([distribution[key] for key in distribution])
        print(['%s: %d %f' % (label, distribution[label], float(
            distribution[label]) / n) for label in distribution])
        return

    def _get_node_name(self, col, row):
        return '%d_%d' % (col, row)

    def _load_annotations(self, annotation_file_path):
        parser = AnnotationParser(annotation_file_path)
        return parser.get_annotaions_for_all_files()

    def _load_sheet_data(self, sheet_folder):
        parser = SheetParser(sheet_folder)
        file_set = {
            x for (x, y) in self.sheet_set} if self.sheet_set is not None else None
        return parser.parse_all_sheets(file_set=file_set)


def create_arg_parser():
    parser = ArgumentParser("feature_generator",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve',
                            description='''Extracts features for the DECO classifier.''')
    parser.add_argument('-s', '--sheet-folder',
                        help="path to folder with spreadsheets", required=True, nargs=1)
    parser.add_argument('-a', '--annotation-file',
                        help="path to file with annotations of the spreadsheets in the spreadsheet folder", required=True, nargs=1)
    parser.add_argument('-f', '--features-file',
                        help="path to file with features of cells of the annotated spreadsheets", required=True, nargs=1)
    parser.add_argument('-d', '--features-description',
                        help="path to file with a list of features that should be extracted from the feature file and their data types", required=True, nargs=1)
    parser.add_argument('-m', '--max-sheet-size',
                        help="maximal number of cells in a spreadsheet (larger sheets are not considered)", default=[DEFAULT_MAX_SHEET_SIZE], nargs=1)
    parser.add_argument('-re', '--remove-empty-cells', nargs='?', const=True, default=False,
                        help="remove cells with an empty value")
    parser.add_argument('-rf', '--remove-no-feature-cells', nargs='?', const=True, default=False,
                        help="remove cells not present in the feature file")
    parser.add_argument('-o', '--output-path',
                        help="output path for pickle file", required=True, nargs=1)
    parser.add_argument('-sl', '--simple-labels', nargs='?', const=True, default=False,
                        help="Usually 7 labels are extracted, however a simpler classification activated by this flag only annotates 3 labels")

    return parser


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()

    print('Create feature generator object ...')
    feature_generator = FeatureGenerator(
        annotation_file_path=args.annotation_file[0], sheet_folder=args.sheet_folder[0],
        feature_file_path=args.features_file[0], feature_description_file_path=args.features_description[0], max_sheet_size=int(args.max_sheet_size[0]))
    print('Create labels ...')
    feature_generator.create_labels(simple=args.simple_labels)
    print('Create features ...')
    deco_default_vector = 'none' if args.remove_no_feature_cells else 'zero'
    feature_generator.get_features(deco_default_vector=deco_default_vector)
    if args.remove_empty_cells:
        print('Remove empty cells ...')
        feature_generator.remove_empty_cells()
    if args.remove_no_feature_cells:
        print('Remove no feature cells ...')
        feature_generator.remove_no_feature_cells()
    print('Create graphs ...')
    feature_generator.get_graph()
    print('Export features')
    feature_generator.export_feature_data(args.output_path[0])
    return


if __name__ == "__main__":
    main()
