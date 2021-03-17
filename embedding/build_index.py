
import re
import gzip
import codecs
import ujson as json
from collections import defaultdict
from itertools import combinations
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter

import utils


def load_dump(path, size=float('inf')):
    f = gzip.open(path, 'rb')
    count = 0
    result_tables = list()
    meta_data = f.readline().decode('utf-8')
    line = f.readline().decode('utf-8')
    while line:
        count += 1
        if count % 10000 == 0:
            print('Processed', count, 'tables')
        if count > size:
            break
        table = json.loads(line)
        result_tables.append(table)
        line = f.readline().decode('utf-8')
    f.close()
    return result_tables, meta_data


def create_table_index(tables):
    table_header = defaultdict(lambda: defaultdict(int))
    header_table = defaultdict(lambda: defaultdict(int))
    for table in tables:
        for attribute in table['relation']:
            header, data = utils.split_attribute(attribute)
            if header != None:
                table_header[table['url']][header] += 1
                header_table[header][table['url']] += 1
    return table_header, header_table


def create_concept_index(tables):
    header_data = defaultdict(lambda: defaultdict(int))
    data_header = defaultdict(lambda: defaultdict(int))
    for table in tables:
        for attribute in table['relation']:
            header, data = utils.split_attribute(attribute)
            if header != None:
                for point in data:
                    if point != None:
                        if point == header:
                            continue  # exclude data elements which are
                            # equivalent with header term and therefore
                            # probably terms of intermediate headers
                        data_header[point][header] += 1
                        header_data[header][point] += 1
    return header_data, data_header

def save_indexes_to_file(indexes, output_file):
    f = gzip.open(output_file, 'w')
    writer = codecs.getwriter('utf-8')
    f2 = writer(f)
    json.dump(indexes, f2)
    f.close()
    return


def create_indexes(tables, meta_data):
    print('Create table index ...')
    table_header, header_table = create_table_index(tables)
    print('Create concept index ...')
    header_data, data_header = create_concept_index(tables)
    print('Finished index creation')
    all_indexes = dict({
        'dump_info': meta_data,
        'table->header': table_header,
        'header->table': header_table,
        'header->data': header_data,
        'data->header': data_header,
    })
    return all_indexes


def create_arg_parser():
    parser = ArgumentParser("build_index",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve',
                            description='''This script builds index structures for a web table corpus.
                            It takes a dump as an input and build certain indexes which are stored in a json formated file''')
    parser.add_argument('-i', '--input',
                        help="dump that should be used for index creation", required=True, nargs=1)
    parser.add_argument('-o', '--output',
                        help="path to file where index should be stored", required=True, nargs=1)
    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    input_path = args.input[0]
    output_path = args.output[0]

    print('Read tables ...')
    tables, meta_data = load_dump(input_path)
    indexes = create_indexes(tables, meta_data)
    print('Store indexes to file ...')
    save_indexes_to_file(indexes, output_path)
    print('Done')
    return


if __name__ == "__main__":
    main()
