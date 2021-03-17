# Table Layout Classification

This tool is able to classify layout types of web tables: *Relation*, *Matrix*, *Entity*, and *Other*.

The classifier performs a supervised training.
It can be trained on an [SQLite corpus](https://wwwdb.inf.tu-dresden.de/misc/web-table-embeddings/labeled_layouts/data.db.gz) with 5700 tables.
To train a random forest classifier, features for the tables can be obtained from an arff file.
For the SQLite corpus, you can use this [arff file](https://wwwdb.inf.tu-dresden.de/misc/web-table-embeddings/labeled_layouts/selected_features.arff) with features obtained by the [DWTC-Extractor](https://github.com/JulianEberius/dwtc-extractor).

## Execute classifier

The classifier can be trained and executed with the web table embedding model as follows:

```
python3 table_layout_classification/evaluate_classifier.py -i path/to/sqlite-corpus/data.db -a data/structural_features.arff -o data/table_layout_classification_results_web_table_model.txt -w path/to/web_table_embedding_model.bin -m web_table_model -s web_table_model -h 50

```

Alternatively, one can use a [fastText word embedding model](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip):

```
python3 table_layout_classification/evaluate_classifier.py -i path/to/sqlite-corpus/data.db -a data/structural_features.arff -o data/table_layout_classification_results_fastText.txt -f ath/to/fastText/model.bin -m word_embedding_model -s word_embedding_model -h 50
```
