# DECO Classifier #

This directory contains an implementation of a GGN-based classifier that uses web table embeddings to classify spreadsheets of the DECO dataset.

### Setup ###

First, one has to provide a dataset with labeled spreadsheets.
We used the [DECO](https://wwwdb.inf.tu-dresden.de/misc/deco/dataset.zip) spreadsheet dataset.

In addition to the dataset, a set of [features](https://drive.google.com/file/d/1_xOEBfryuFzOUU6bHRZOJl8m1bYko4Wg/view?usp=sharing) for training a random forest classifier can be downloaded.

Annotations for the spreadsheets can be found in the repository of the [Annotation Exporter](https://github.com/ddenron/annotations_exporter).

To set up the classification scripts, one has to install all necessary python packages.
All packages can be installed via pip.
We used the [DGL](https://www.dgl.ai/) framework to implement the graph neural network.
Here, the framework has to be configured to use the mxnet backend.

### Run classification ###
First, a python pickle file with the features can be constructed by executing the following command:

```
mkdir data && python3 feature_generator.py -a deco_annotation_file -s spreadsheet_folder -f deco_features_file -d deco_feature_descriptions -o data/features.pkl -m 100 -re -rf -sl

```

Afterward, you can execute the classification with the features file and a web table embedding model:

```
python3 evaluate_classifier.py -i data/features.pkl -t web-table -e web_table_embedding_model.txt -f embeddings deco -o data/output.sqlite -it 10 -c gnn random-forest
```

During the execution, the classification script creates an SQLite database with the evaluation results in the `data` folder.

