
import os
import json
import numpy as np
import pandas as pd
import sqlite3
from collections import defaultdict
from collections import Counter


class TableExport:
    def __init__(self, node_ids, pred, dgl_graph, node_attributes, features, labels, scores, labeling=None):
        node_ids = node_ids.asnumpy()
        self.pred = {node_ids[id]: pred[node_ids[id]]
                     for id in range(len(node_ids))}
        self.labeling = None
        if labeling is not None:
            self.labeling = labeling.asnumpy()
            self.labeling = {node_ids[id]: labeling[id].asnumpy()
                             for id in range(len(node_ids))}
        self.dgl_graph = dgl_graph
        self.features = features
        self.node_attributes = node_attributes
        self.get_label = {pos: label for pos, label in enumerate(labels)}
        self.tables = self._construct_tables()
        self.scores = scores
        print('All scores:', self.scores)

    def export_tables_as_json(self, filename):
        f = open(filename, 'w')
        json.dump(self.tables, f)
        return

    def export_tables_as_sqlite(self, filename, iteration):
        SCHEMA_QUERIES = (
            "CREATE TABLE IF NOT EXISTS sheets (id INTEGER PRIMARY KEY, iteration INTEGER, filename TEXT , sheetname TEXT , data TEXT, prediction_diversity REAL, label_diversity REAL)",
            "CREATE TABLE IF NOT EXISTS scores (id INTEGER PRIMARY KEY, iteration INTEGER, metric TEXT, score REAL)"
        )

        INSERT_SHEET_QUERY = "INSERT INTO sheets (iteration, filename, sheetname, data, prediction_diversity, label_diversity) VALUES (?, ?, ?, ?, ?, ?)"
        INSERT_SCORE_QUERY = "INSERT INTO scores (iteration, metric, score) VALUES (?, ?, ?)"

        con = sqlite3.connect(filename)
        cur = con.cursor()
        # create schema
        for query in SCHEMA_QUERIES:
            cur.execute(query)
        # insert sheet data
        for key, sheet_data in self.tables.items():
            filename = sheet_data['filename']
            sheetname = sheet_data['sheetname']
            data = sheet_data['table']
            prediction_diversity = sheet_data['prediction_diversity']
            label_diversity = sheet_data['label_diversity']
            cur.execute(INSERT_SHEET_QUERY, (iteration, filename, sheetname,
                                             data, prediction_diversity, label_diversity))
        # insert evaluation scores
        for metric, score in self.scores.items():
            cur.execute(INSERT_SCORE_QUERY, (iteration, metric, score))
        con.commit()
        return

    def clear_sqlite_db(self, filename):
        if os.path.exists(filename):
            os.remove(filename)

    def _construct_tables(self):
        # construct dictionays
        print('Start to construct sheet data for export ...')
        tables = defaultdict(dict)
        for dgl_node_id, node_id in enumerate(self.dgl_graph.ndata['node_id'].asnumpy()):
            attributes = self.node_attributes[node_id]
            if dgl_node_id in self.pred:
                sheet_key = (attributes['filename'], attributes['sheetname'])
                cell_label = self.get_label[int(
                    self.labeling[dgl_node_id])] if self.labeling is not None else None
                cell_prediction = self.get_label[int(
                    np.argmax(self.pred[dgl_node_id].asnumpy()))]
                cell_content = self.features[sheet_key][(
                    attributes['col'], attributes['row'])]['content']
                tables[sheet_key][(attributes['col'], attributes['row'])] = (
                    cell_content, cell_prediction, cell_label)
        # construct matrixes from dictionaries
        result = dict()
        for sheet_key in tables:
            pred_distribution = Counter(
                [pred for (content, pred, label) in tables[sheet_key].values()])
            pred_diversity = 1.0 - \
                (pred_distribution.most_common()[
                 0][1] / sum(pred_distribution.values()))
            label_distribution = Counter(
                [label for (content, pred, label) in tables[sheet_key].values()])
            label_diversity = 1.0 - \
                (label_distribution.most_common()[
                 0][1] / sum(label_distribution.values()))
            result['-'.join(sheet_key)] = {
                'filename': sheet_key[0],
                'sheetname': sheet_key[1],
                'table': self._table_dict2matrix(tables[sheet_key]),
                'prediction_diversity': pred_diversity,
                'label_diversity': label_diversity
            }
        return result

    def _table_dict2matrix(self, table_dict):
        n_cols = max([x[0] for x in table_dict.keys()])
        n_rows = max([x[1] for x in table_dict.keys()])
        columns = []
        for i in range(n_cols):
            col = []
            for j in range(n_rows):
                if (i, j) in table_dict:
                    col.append('::'.join([str(x)
                                          for x in table_dict[(i, j)]]))
                else:
                    col.append('')
            columns.append(col)
        df = pd.DataFrame(data=columns)
        return df.to_csv()
