import gzip
import lightrdf
import re
import pickle
import random

from collections import defaultdict

CLASSES_FILENAME = 'yago-wd-class.nt.gz'
LABELS_FILENAME = 'yago-wd-labels.nt.gz'
TYPES_FILENAME = 'yago-wd-simple-types.nt.gz'
SCHEMA_ORG_FILENAME = 'schemaorg-current-http.nt'
PICKLE_FILENAME = 'taxonomy.pkl'

CLASS_BLACKLIST = ['http://schema.org/Thing']


class Taxonomy:
    def __init__(self, taxonomy_folder):
        self.taxonomy_folder = taxonomy_folder
        self.label_lookup = None
        self.schema_label_lookup = None
        self.is_a = None

    def construct_taxonomy(self):
        r_language = re.compile('@[0-9a-z\-]*$')
        r_label = re.compile('^\".*\"')

        f_classes = gzip.open(self.taxonomy_folder + CLASSES_FILENAME, 'rb')
        f_labels = gzip.open(self.taxonomy_folder + LABELS_FILENAME, 'rb')
        f_types = gzip.open(self.taxonomy_folder + TYPES_FILENAME, 'rb')
        f_schema_labels = open(self.taxonomy_folder +
                               SCHEMA_ORG_FILENAME, 'rb')
        classes = lightrdf.RDFDocument(
            f_classes, parser=lightrdf.nt.PatternParser)
        labels = lightrdf.RDFDocument(
            f_labels, parser=lightrdf.nt.PatternParser)
        types = lightrdf.RDFDocument(f_types, parser=lightrdf.nt.PatternParser)
        schema_labels = lightrdf.RDFDocument(
            f_schema_labels, parser=lightrdf.nt.PatternParser)

        # Parse all labels
        self.label_lookup = dict()
        for i, (s, p, o) in enumerate(labels.search_triples(None, 'http://www.w3.org/2000/01/rdf-schema#label', None)):
            language_tags = r_language.findall(o)
            if language_tags[0] == '@en':
                label = r_label.findall(o)[0][1:-1]
                self.label_lookup[s] = label
            if (i % 1000000 == 0) and (i > 0):
                print('Parsed', str(i / 1000000) + 'M triples')

        # Parse types
        self.is_a = defaultdict(set)
        for (s, p, o) in types.search_triples(None, 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type', None):
            self.is_a[s].add(o)
        to_remove = (set(self.is_a.keys()) - set(self.label_lookup.keys()))
        for s in to_remove:
            del self.is_a[s]

        # Parse Schema Labels
        self.schema_label_lookup = dict()
        for (s, p, o) in schema_labels.search_triples(None, 'http://www.w3.org/2000/01/rdf-schema#label', None):
            self.schema_label_lookup[s] = o[1:-1]
        # Resolve camel case
        resolve = dict()
        for value in self.schema_label_lookup.values():
            resolve[value] = self._resolve_camel_case(value)
        for key in self.schema_label_lookup:
            self.schema_label_lookup[key] = resolve[self.schema_label_lookup[key]]

        # Parse class hierarchy
        for (s, p, o) in classes.search_triples(None, 'http://www.w3.org/2000/01/rdf-schema#subClassOf', None):
            self.is_a[s].add(o)

        return

    def save_taxonomy(self):
        taxonomy = {
            "label-lookup": self.label_lookup,
            "schema-label-lookup": self.schema_label_lookup,
            "is-a": self.is_a
        }
        f_out = open(self.taxonomy_folder + PICKLE_FILENAME, 'wb')
        pickle.dump(taxonomy, f_out)

    def load_taxonomy(self):
        f = open(self.taxonomy_folder + PICKLE_FILENAME, 'rb')
        taxonomy = pickle.load(f)
        self.label_lookup = taxonomy['label-lookup']
        self.schema_label_lookup = taxonomy['schema-label-lookup']
        self.is_a = taxonomy['is-a']
        return

    def sample_links(self, num_positive, num_negative):
        instances = list(self.is_a.keys())
        all_classes = list(set.union(*self.is_a.values()) -
                           set(CLASS_BLACKLIST))

        # get positive samples
        positive_samples = []
        while len(positive_samples) < num_positive:
            p = random.randint(0, len(instances) - 1)
            if instances[p] not in self.label_lookup:
                continue
            inst_label = self.label_lookup[instances[p]]
            classes = self.is_a[instances[p]] - set(CLASS_BLACKLIST)
            class_labels = [self.schema_label_lookup[c]
                            for c in classes if c in self.schema_label_lookup]
            if len(class_labels) == 0:
                continue
            class_label = random.choice(class_labels)
            positive_samples.append((inst_label, class_label, 1))

        # get negative samples
        negative_samples = []
        while len(negative_samples) < num_negative:
            p = random.randint(0, len(instances) - 1)
            inst = instances[p]
            if inst not in self.label_lookup:
                continue
            # get all classes of inst
            classes_of_inst = set()
            classes_of_inst.update(self.is_a[inst])
            frontier = set()
            frontier.update(self.is_a[inst])
            while len(frontier) > 0:
                next = frontier.pop()
                new_classes = (self.is_a[next] - {next}) - classes_of_inst
                classes_of_inst.update(new_classes)
                frontier.update(new_classes)
            # sample invalid class
            c = None
            while True:
                p = random.randint(0, len(all_classes) - 1)
                c = all_classes[p]
                if c not in self.schema_label_lookup:
                    continue
                if c not in classes_of_inst:
                    break
            # get labels
            inst_label = self.label_lookup[inst]
            class_label = self.schema_label_lookup[c]
            negative_samples.append((inst_label, class_label, 0))

        # shuffle samples
        samples = positive_samples + negative_samples
        random.shuffle(samples)
        return samples

    def _resolve_camel_case(self, value):
        result = value[0]
        for i in range(1, len(value)):
            if value[i].isupper():
                result += ' '
            result += value[i]
        return result
