from zipfile import ZipFile

class FeatureParser:
    def __init__(self, filename):
        self.FEATURES_FILENAME = 'features_enron_854.csv'
        self.archive = ZipFile(filename)
        self.features = self.extract_features()

    def extract_features(self):
        features_file = self.archive.open(FEATURES_FILENAME, 'r')
