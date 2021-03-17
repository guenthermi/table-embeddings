import sqlite3
import json


class LabeledCorpusInterface:
    """Interface for labeled dwtc corpus (SQLite DB)
    """

    def __init__(self, file_path):
        self.con = sqlite3.connect(file_path)
        self.cur = self.con.cursor()

    def get_all_urls(self):
        URL_QUERY = "SELECT url FROM 'table'"
        url_list = [x for (x,) in self.cur.execute(URL_QUERY)]
        return url_list

    def get_table_data(self):
        QUERY = "SELECT id, cells FROM 'table' WHERE label IS NOT NULL"
        table_data = dict()
        for id, cell_data in self.cur.execute(QUERY):
            table_data[id] = json.loads(cell_data)
        return table_data

    def get_table_by_id(self, id):
        query = "SELECT cells FROM 'table' WHERE id = %d" % (int(id),)
        [(cells,)] = self.cur.execute(query)
        data = json.loads(cells)
        return data


    def get_labels(self):
        QUERY = "SELECT id, label FROM 'table' WHERE label IS NOT NULL"
        label_data = dict()
        for id, label in self.cur.execute(QUERY):
            label_data[id] = label
        return label_data
