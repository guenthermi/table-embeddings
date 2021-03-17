import sys
import pandas as pd


class AnnotationParser:
    """Interface for anntotation files created by
    Annotation Exporter (https://github.com/ddenron/annotations_exporter)

    Attributes:
        filename: path to anntotations file
    """

    def __init__(self, filename):
        self.data = pd.read_csv(filename)
        return

    def get_annotaions_for_file(self, filename):
        """Returns annotations of a specific spreadsheet file.
        """
        return self.data[self.data['FileName'] == filename]

    def get_annotaions_for_all_files(self):
        """Returns annotations of all spreadsheet files in the annotation file.
        """
        result = dict()
        file_name_groups = self.data.groupby('FileName')
        for i, file_name in enumerate(file_name_groups.groups):
            result[file_name] = dict()
            df_file = file_name_groups.get_group(file_name)
            sheet_name_groups = df_file.groupby('SheetName')
            for sheet_name in sheet_name_groups.groups:
                df_sheet = sheet_name_groups.get_group(sheet_name)
                result[file_name][sheet_name] = df_sheet
        return result
