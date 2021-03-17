
import sys
import _thread
import xlrd
from pathlib import Path
import time


class SheetParser:
    def __init__(self, sheet_folder_path):
        self.sheet_folder_path = Path(sheet_folder_path)
        return

    def parse_all_sheets(self, file_set=None):
        sheets = dict()
        c = 0

        def thread_func(filename, dest):
            dest.append((filename, self._parse_sheets_from_file(filename)))
            return
        parsed_sheets = []
        print('Start threads ...')
        files = list(self.sheet_folder_path.glob('*'))
        number_of_files = 0
        for file_path in files:
            if (file_set is None) or (file_path.name in file_set):
                number_of_files += 1
                _thread.start_new_thread(
                    thread_func, (file_path.name, parsed_sheets))
        print('Parsing', number_of_files, '/', len(files), 'sheet files ...')
        while len(parsed_sheets) < number_of_files:
            time.sleep(0.1)
            print('\rLoad Sheets %05.2f %%' %
                  (float(len(parsed_sheets)) * 100 / number_of_files), end='')
        print('')
        sheets = dict(parsed_sheets)
        return sheets

    def _parse_sheets_from_file(self, filename):
        file_path = self.sheet_folder_path.joinpath(filename)
        book = xlrd.open_workbook(file_path)
        return book
