import os
import sys
import itertools
import ujson as json
import networkx as nx
from scipy import sparse
from collections import defaultdict
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import random

import utils


class TermsIndex:
    def __init__(self, indexes):
        print('Create termlist table_terms ...')
        self.table_terms = set()
        if indexes['table->header'] is not None:
            self.table_terms = list(indexes['table->header'].keys())
        self.table_terms_lookup = TermsIndex.create_terms_lookup(
            self.table_terms)

        print('Create termlist header_terms ...')
        self.header_terms = set()
        if indexes['table->header'] is not None:
            for key in self.table_terms:
                self.header_terms.update(indexes['table->header'][key].keys())
        self.header_terms = list(self.header_terms)
        self.header_terms_lookup = TermsIndex.create_terms_lookup(
            self.header_terms)

        print('Create termlist data_terms ...')
        self.data_terms = set()
        if indexes['header->data'] is not None:
            for key in indexes['header->data']:
                self.data_terms.update(indexes['header->data'][key].keys())
        self.data_terms = list(self.data_terms)
        self.data_terms_lookup = TermsIndex.create_terms_lookup(
            self.data_terms)
        return

    @staticmethod
    def create_terms_lookup(termlist):
        lookup = dict()
        for i, elem in enumerate(termlist):
            lookup[elem] = i
        return lookup


class WebTableGraph:
    def __init__(self, indexes, terms_index):
        self.indexes = indexes
        self.terms_index = terms_index
        self.term_list = None
        self.G = None
        return

    def output_graph_data(self, output_path):
        f_terms = open(output_path + '-terms.txt', 'w')
        nx.write_weighted_edgelist(self.G, output_path + '-edges.txt')
        for term in self.term_list:
            f_terms.write(term + '\n')
        f_terms.close()
        return


class HeaderDataGraph(WebTableGraph):

    def __init__(self, indexes, termlists, min_headers):

        super(HeaderDataGraph, self).__init__(indexes, termlists)

        REMOVE_RARE_TERMS = True
        REMOVE_BIDIRECTIONALS = True

        if REMOVE_RARE_TERMS:
            print('remove rare terms ...')
            self._remove_rare_terms(min_headers)

        if REMOVE_BIDIRECTIONALS:
            print('remove bidirectionals ...')
            self._remove_bidirectionals()

        print('create graph ...')
        offset = len(termlists.header_terms_lookup)
        self.term_list = ['h#' + x for x in termlists.header_terms
                          ] + ['d#' + x for x in termlists.data_terms]
        self.G = nx.Graph()
        for header in indexes['header->data']:
            for data_term in indexes['header->data'][header]:
                header_id = termlists.header_terms_lookup[header]
                data_term_id = termlists.data_terms_lookup[data_term] + offset
                self.G.add_edges_from(
                    [(header_id, data_term_id, {'weight': indexes['header->data'][header][data_term]})])
        return

    def _remove_rare_terms(self, min_headers):
        header_data = self.indexes['header->data']
        data_header = self.indexes['data->header']
        to_remove_data_header = list()
        count = 0
        for data_elem in data_header:
            if len(data_header[data_elem]) < min_headers:
                for header in data_header[data_elem]:
                    del header_data[header][data_elem]
                to_remove_data_header.append(data_elem)
        count = 0
        print('Process to_remove_data_header ...')
        for data_elem in to_remove_data_header:
            count += 1
            if count % 100000 == 0:
                print('Removed', count, 'elements')
            del data_header[data_elem]
        return

    def _remove_bidirectionals(self):
        header_data = self.indexes['header->data']
        data_header = self.indexes['data->header']
        to_remove_data_header = list()
        to_remove_header_data = list()

        for first_elem in header_data:
            if first_elem in data_header:
                for second_elem in header_data[first_elem]:
                    if second_elem in data_header[first_elem]:
                        freq_header_data = header_data[first_elem][second_elem]
                        freq_data_header = data_header[first_elem][second_elem]
                        if freq_header_data > freq_data_header:
                            to_remove_data_header.append(
                                (first_elem, second_elem))
                        elif freq_header_data < freq_data_header:
                            to_remove_header_data.append(
                                (first_elem, second_elem))
        # remove elements
        for (first_elem, second_elem) in to_remove_data_header:
            del data_header[first_elem][second_elem]
            if len(data_header[first_elem]) == 0:
                del data_header[first_elem]
        for (first_elem, second_elem) in to_remove_header_data:
            del header_data[first_elem][second_elem]
            if len(header_data[first_elem]) == 0:
                del header_data[first_elem]
        return


def create_arg_parser():
    parser = ArgumentParser("graph_generation",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve',
                            description='''This tool creates a graph representation
                                based on the index data created with build_index.py.
                                The graph is stored as edgelist of ids.
                                In order to resolve the meaning of the ids, a termlist is created.''')
    parser.add_argument('-i', '--index-path',
                        help="path to file with index data", required=True, nargs=1)
    parser.add_argument('-c', '--config',
                        help="files with configuration of the graph and output destination", required=True, nargs='*')
    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    configs = args.config

    print('Load indexes ...')
    indexes = utils.load_index_file(args.index_path[0])
    print('Create termlists ...')
    term_lists = TermsIndex(indexes)

    for config_path in configs:
        f_config = open(config_path, 'r')
        config = json.load(f_config)

        print('Create header<->data graph ...')
        graph = HeaderDataGraph(indexes, term_lists, config['min_header_count'])

        graph.output_graph_data(config['output_path'])

    return


if __name__ == "__main__":
    main()
