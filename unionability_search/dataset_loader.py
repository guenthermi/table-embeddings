import sqlite3
import re

BASE_FILENAME = 'base.sqlite'
GROUNDTRUTH_FILENAME = 'groundtruth.sqlite'
BENCHMARK_FILENAME = 'benchmark.sqlite'

RETURN_TYPE = 'list'

RE_NOT_EMPTY = re.compile('[0-9a-zA-Z]')


class DatasetLoader:
    def __init__(self, path):
        self.dataset_path = path
        self.alignmentsA, self.alignmentsB = self._load_alignments()
        self.columns, self.all_text_values = self._load_columns(
            self.alignmentsA, self.alignmentsB)
        return

    def _connect_to_database(self, file_path):
        con = sqlite3.connect(file_path)
        cur = con.cursor()
        return con, cur

    def _load_alignments(self):
        QUERY_GET_ALIGNMENTS = "SELECT query_table, candidate_table, query_col_name, candidate_col_name FROM att_groundtruth"

        result = dict()
        inverse_result = dict()

        con, cur = self._connect_to_database(
            self.dataset_path + GROUNDTRUTH_FILENAME)

        cur.execute(QUERY_GET_ALIGNMENTS)
        for (query, candidate, query_col, candidate_col) in cur.fetchall():
            query = query.replace('.csv', '')
            candidate = candidate.replace('.csv', '')
            if (query, query_col) not in result:
                result[(query, query_col)] = set()
            result[(query, query_col)].add((candidate, candidate_col))
            if (candidate, candidate_col) not in inverse_result:
                inverse_result[(candidate, candidate_col)] = set()
            inverse_result[(candidate, candidate_col)].add((query, query_col))

        return result, inverse_result

    def _load_columns(self, alignments, alignments_inv):
        QUERY_GET_TABLE = "pragma table_info('%s')"
        # result
        columns = dict()
        all_text_values = dict()
        # get all relevant tables
        present_tables = set([x[0] for x in alignments]) | set(
            [x[0] for x in alignments_inv])
        # retrieve all columns from the relevant tables
        con, cur = self._connect_to_database(
            self.dataset_path + BENCHMARK_FILENAME)
        for table_name in present_tables:
            cur.execute(QUERY_GET_TABLE % (table_name,))
            column_data = cur.fetchall()
            if len(column_data) < 1:
                print('ERROR: missing table:', table_name)
            for (_, col_name, col_type, _, _, _) in column_data:
                if ((table_name, col_name) in alignments) or (
                        (table_name, col_name) in alignments_inv):
                    columns[(table_name, col_name)] = set()
                    text_values = self._get_text_values(
                        con, cur, table_name, col_name)
                    for text_value in text_values:
                        if text_value not in all_text_values:
                            if text_value is None:
                                print('ERROR')
                                quit()
                            all_text_values[text_value] = len(all_text_values)
                        columns[(table_name, col_name)].add(
                            all_text_values[text_value])
        all_text_values_reverse = dict()
        for value in all_text_values:
            all_text_values_reverse[all_text_values[value]] = value
        return columns, all_text_values_reverse

    def _get_text_values(self, con, cur, table_name, column_name,
                         return_type=RETURN_TYPE):
        QUERY_GET_TABLE = "SELECT \"%s\" FROM %s"
        cur.execute(QUERY_GET_TABLE % (column_name, table_name))
        if return_type == 'set':
            return set([x[0] for x in cur.fetchall() if (x[0] is not None) and (
                RE_NOT_EMPTY.match(str(x[0])) is not None)])
        elif return_type == 'list':
            return list([x[0] for x in cur.fetchall() if (x[0] is not None) and (
                RE_NOT_EMPTY.match(str(x[0])) is not None)])
        else:
            print('ERROR: Unknown return type:', return_type)
            return

    def get_alignments(self):
        return self.alignmentsA, self.alignmentsB

    def get_column(self, table_name, column_name):
        col = self.columns[(table_name, column_name)]
        return [self.all_text_values[x] for x in col]
