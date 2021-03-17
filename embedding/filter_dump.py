
import os
import sys
import ujson as json
import tldextract
import gzip
import logging
from datetime import datetime
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter


class TableFilter:

    def __init__(self, corpus_path):
        self.corpus_path = corpus_path  # directory of web table corpus with muliple files
        # Get all filenames of the corpus
        self.file_names = [x for x in os.listdir(
            self.corpus_path) if not os.path.isdir(os.path.join(self.corpus_path, x))]
        self.result_tables = list()

    def table_filter_func(self):
        # This function has to be impolemented by the filter
        return False

    def apply_filter(self):
        self.result_tables = list()
        count = 0
        for i, filename in enumerate(self.file_names):
            f = gzip.open(self.corpus_path + filename, 'rb')
            line = f.readline().decode('utf-8')
            print('Process %s (%d/%d)' %
                  (filename, i + 1, len(self.file_names)))
            while line:
                data = json.loads(line)
                if self.table_filter_func(data):
                    self.result_tables.append(line)
                try:
                    line = f.readline().decode('utf-8')
                except:
                    print('Problem with parsing line in ' + filename)
                    break
            f.close()
        return

    def output_tables(self, meta_data, output_filename):
        f = gzip.open(output_filename, 'w')
        f.write((json.dumps(meta_data) + '\n').encode('utf-8'))
        for line in self.result_tables:
            f.write((line).encode('utf-8'))
        f.close()
        return


class WikiTableFilter(TableFilter):

    def __init__(self, corpus_path, config):
        super(WikiTableFilter, self).__init__(corpus_path)

    def table_filter_func(self, table):
        if ((table['hasHeader'] == True) and
                (table['url'].find('en.wikipedia.org') != -1) and
                ((table['headerPosition'] == 'FIRST_ROW') or (table['headerPosition'] == 'FIRST_COLUMN')) and
                (len(table['relation'][0]) > 2)):
            return True
        else:
            return False


class EnglishTableFilter(TableFilter):

    def __init__(self, corpus_path, config, has_header=True):
        super(EnglishTableFilter, self).__init__(corpus_path)
        self.has_header = has_header
        self.min_row_count = config['min_row_count']
        logging.getLogger("tldextract").setLevel(logging.CRITICAL)

    def table_filter_func(self, table):
        if (((table['headerPosition'] == 'FIRST_ROW') or (table['headerPosition'] == 'FIRST_COLUMN')) and
                (len(table['relation'][0]) >= self.min_row_count)):
            url_suffix=tldextract.extract(table['url']).suffix
            if ((table['url'].find('en.wikipedia.org') != -1) or (url_suffix in {'us', 'uk', 'co.uk', 'org.uk', 'gov', 'com'})):
                if ((table['hasHeader'] == True) and ((table['headerPosition'] == 'FIRST_ROW') or (table['headerPosition'] == 'FIRST_COLUMN'))):
                    return True
                else:
                    return not self.has_header
            else:
                return False
        else:
            return False


def create_arg_parser():
    parser=ArgumentParser("filter_dump",
                            formatter_class = ArgumentDefaultsHelpFormatter,
                            conflict_handler = 'resolve',
                            description = 'Create a filtered web table dump of the DWTC corpus')
    parser.add_argument('-c', '--config',
                        help="file with configuration of input, output destination, and filter criteria", required=True, nargs=1)

    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    f_config = open(args.config[0], 'r')

    config = json.load(f_config)
    corpus_path = config['source_dump']
    dump_type = config['dump_type']
    output_path = config['output_file']

    FILTERS = {
        'wiki-dump': WikiTableFilter,
        'english-dump': EnglishTableFilter
    }

    if config['dump_type'] in FILTERS:
        filter = FILTERS[dump_type](corpus_path, config)
        filter.apply_filter()
        meta_data = config
        meta_data['time_stamp'] = datetime.now().ctime()
        filter.output_tables(meta_data, output_path)
    else:
        print('ERROR: Unknown dump type %s' % (dump_type,), file=sys.stderr)


if __name__ == "__main__":
    main()
