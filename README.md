# Pretrained Web Table Embeddings

This repository contains tools for training Web table embedding with word embedding techniques.
Those models can generate embeddings for schema terms and instance data terms making them especially useful for representing schema and class information as well as for ML tasks on tabular text data.
Furthermore, this repository contains links to pre-trained web table models and the code for several tasks the models can be used for.

## Install Package

If you want to install the package to encode text (from tables) into embedding representations, you can run

```
pip install .
```

and load a pre-trained model as follows:

```
from table_embeddings import TableEmbeddingModel
model = TableEmbeddingModel.load_model('ddrg/web_table_embeddings_combo64')

embedding = model.get_header_vector('headline')
```

For installing all dependencies to run the evaluation tasks you can run:

```
pip install ".[full]"
```

## Embedding Training

This repository provides tools for training four different types of Web table embedding models: *W-base*, *W-row*, *W-tax*, and *W-combo*.
For pre-training those embedding models the [DWTC Web Table Corpus](https://wwwdb.inf.tu-dresden.de/misc/dwtc/]) can be used.
All modules required to run the python scripts in this repository can be installed via pip.

#### Download DWTC Dump

The corpus can be downloaded as follows:
```
for i in $(seq -w 0 500); do wget http://wwwdb.inf.tu-dresden.de/misc/dwtc/data_feb15/dwtc-$i.json.gz -P data/; done
```

### Filter Dump

The DWTC dump can be filtered with `embedding/filter_dump.py` and `embedding/filter_columns.py` to create a dump containing only columns of English tables with a table header.
You may adjust the path of the DWTC corpus in `config/dump_filter.json`.

```
python3 embedding/filter_dump.py -c config/dump_filter.json
python3 embedding/filter_columns.py -c config/column_filter.json
```


### Construct Graph Representation

To train *W-tax* and *W-combo* embedding models, a header-data term graph needs to be constructed.
First, an index file is constructed:

```
python3 embedding/build_index.py -i data/column_dump.json.gz -o data/indexes.json.gz
```

Afterward, the graph can be constructed:

```
python3 embedding/graph_generation.py -i data/indexes.json.gz -c config/header_data_graph_config.json
```

### Training of Embedding Models

To run the actual embedding training, one can execute `embedding/fasttext_web_table_embeddings.py` with one of the embedding configuration files in the config folder:

```
python3 embedding/fasttext_web_table_embeddings.py -c config/embedding_config_combo.json -o data/combo_model.bin -w
```


## Pre-Trained Models

Below you can find links to models trained on the DWTC corpus:

| Model Type | Description | Download-Links |
| ---------- | ----------- | -------------- |
| W-tax      | Model of relations between table header and table body | ([64dim](https://wwwdb.inf.tu-dresden.de/misc/web-table-embeddings/web_table_embeddings_tax64.bin.gz), [150dim](https://wwwdb.inf.tu-dresden.de/misc/web-table-embeddings/web_table_embeddings_tax150.bin.gz))
| W-row      | Model of row-wise relations in tables | ([64dim](https://wwwdb.inf.tu-dresden.de/misc/web-table-embeddings/web_table_embeddings_row64.bin.gz), [150dim](https://wwwdb.inf.tu-dresden.de/misc/web-table-embeddings/web_table_embeddings_row150.bin.gz))
| W-combo      | Model of row-wise relations and relations between table header and table body | ([64dim](https://wwwdb.inf.tu-dresden.de/misc/web-table-embeddings/web_table_embeddings_combo64.bin.gz), [150dim](https://wwwdb.inf.tu-dresden.de/misc/web-table-embeddings/web_table_embeddings_combo150.bin.gz))
| W-plain      | Model of row-wise relations in tables without pre-processing | ([64dim](https://wwwdb.inf.tu-dresden.de/misc/web-table-embeddings/web_table_embeddings_plain64.bin.gz), [150dim](https://wwwdb.inf.tu-dresden.de/misc/web-table-embeddings/web_table_embeddings_plain150.bin.gz))

To use the models, you can use the `FastTextWebTableModel.load_model` function in `embedding/fasttext_web_table_embeddings.py`.

## Evaluation

Besides the embedding training, this repository contains the code of four evaluation tasks:

* Representation of instance-of relations found in YAGO (`yago_class_evaluation/`)
* Unionable Table Search (`unionability_search/`)
* Table layout classification on Web tables (`table_layout_classification/`)
* Spreadsheet cell classification (`deco_classifier/`)

A detailed description, how to run the evaluation, is provided in the respective folders.

## References
[Pre-Trained Web Table Embeddings for Table Discovery](https://dl.acm.org/doi/10.1145/3464509.3464892)
```
@inproceedings{gunther2021pre,
  title={Pre-Trained Web Table Embeddings for Table Discovery},
  author={G{\"u}nther, Michael and Thiele, Maik and Gonsior, Julius and Lehner, Wolfgang},
  booktitle={Fourth Workshop in Exploiting AI Techniques for Data Management},
  pages={24--31},
  year={2021}
}
```
