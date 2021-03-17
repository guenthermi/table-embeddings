
import re
import gzip
import ujson as json
from datetime import datetime
from whatthelang import WhatTheLang
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter

import utils

class ColumnFilter:
    def __init__(self, config):
        self.config = config
        self.result_tables = list()

        self.lang_filter = self.config['lang_filter']
        # minimal number of chars in a text value
        self.min_text_value_size = self.config['min_text_value_size']
        # maximal number of chars in a text value
        self.max_text_value_size = self.config['max_text_value_size'] if type(
            self.config['max_text_value_size']) == int else float('inf')
        self.min_col_size = self.config['min_col_size']

        self.re_filter = utils.RE_VALID_STRING

        self.wtl = WhatTheLang()
        return

    def filter_columns(self, table):
        if table['headerPosition'] == 'FIRST_COLUMN':
            table['relation'] = list(zip(*table['relation']))
        columns = list()
        for col in table['relation']:
            col_size = 0
            new_column = []
            for elem in col:
                if (len(elem) < self.min_text_value_size) or (self.re_filter.fullmatch(elem) == None) or (len(elem) > self.max_text_value_size):
                    new_column.append(None)
                else:
                    new_column.append(self._regularize_numbers(self._regularize_special_signs(elem)))
                    col_size += 1
            if col_size >= self.min_col_size:
                columns.append(new_column)
        if len(columns) > 0:
            if self.lang_filter != 'none':
                text = ' '.join(
                    [' '.join([c for c in col if c != None]) for col in columns])
                lang = self.wtl.predict_lang(text)
                if lang != self.lang_filter:
                    return None
            return {
                'relation': columns,
                'url': table['url'],
                'title': table['title']
            }
        return None


    def apply_filter(self):
        BATCH_SIZE = 100000
        size = self.config['max_size'] if type(
            self.config['max_size']) == int else float('inf')
        file_paths = self.config['dump_paths']
        count = 0
        self.result_tables = list()
        self.init_output_file()
        for file_path in file_paths:
            f = gzip.open(file_path, 'rb')
            meta_data = f.readline().decode('utf-8')
            line = f.readline().decode('utf-8')
            while line:
                count += 1
                if count % 10000 == 0:
                    print('Processed', count, 'tables')
                if count > size:
                    break
                data = json.loads(line)
                table = self.filter_columns(data)
                if table != None:
                    self.result_tables.append(table)
                line = f.readline().decode('utf-8')
                if len(self.result_tables) > BATCH_SIZE:
                    self.output_batch()
            f.close()
        self.output_batch()
        self.close_output_file()
        return

    def init_output_file(self):
        self.f_out = gzip.open(self.config['output_file'], 'w')
        meta_data = self.config
        meta_data['time_stamp'] = datetime.now().ctime()
        self.f_out.write((json.dumps(meta_data) + '\n').encode('utf-8'))

    def output_batch(self):
        for table in self.result_tables:
            self.f_out.write((json.dumps(table)+'\n').encode('utf-8'))
        self.result_tables = []

    def close_output_file(self):
        self.f_out.close()

    def output_tables(self):
        # DEPRECATED
        f = gzip.open(self.config['output_file'], 'w')
        meta_data = self.config
        meta_data['time_stamp'] = datetime.now().ctime()
        f.write((json.dumps(meta_data) + '\n').encode('utf-8'))
        for table in self.result_tables:
            f.write((json.dumps(table)+'\n').encode('utf-8'))
        f.close()

    def _regularize_numbers(self, value):
        return utils.RE_REGULARIZE_NUMBERS.sub('@',value)

    def _regularize_special_signs(self, value):
        return utils.RE_REGULARIZE_SPECIAL_SIGNS.sub('*', value)


def create_arg_parser():
    parser = ArgumentParser("filter_columns",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve',
                            description='Takes a dump of web tables and filter the columns of each table by filter criteria')
    parser.add_argument('-c', '--config',
                        help="file with configuration of input, output destination, and filter criteria", required=True, nargs=1)

    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    # Parse config file
    f_config = open(args.config[0], 'r')
    config = json.load(f_config)

    print('Create column filter ...')
    filter = ColumnFilter(config)
    print('Apply column / row filter ...')
    filter.apply_filter()
    print('Done')

if __name__ == "__main__":
    main()
