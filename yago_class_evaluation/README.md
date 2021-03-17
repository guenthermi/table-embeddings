# Class Membership Evaluation on YAGO

Here you can find tools for an evaluation of Web table embeddings and word embeddings with taxonomical data from YAGO.

### Setup

At first, you have to download the English Wikipedia taxonomy files: `yago-wd-class.nt.gz`, `yago-wd-labels.nt.gz`, and `yago-wd-simple-types.nt.gz`, from the [YAGO project page](https://yago-knowledge.org/downloads/yago-4).
Moreover, you need the [schema.org ontology](https://schema.org/docs/developers.html) for the class labels in the N-Triples format.
All files should be stored in a taxonomy folder.
After that, you can run the evaluation by executing the `evaluate_model.py` script:

```
python3 yago_class_evaluation/evaluate_model.py -e path/to/embedding_model -t /path/to/taxonomy-folder/ -et web-table -s 10000 -o data/yago_evaluation_results.json -n 1
```

The results can be plotted using `create_diagrams.py`:

```
python3 yago_class_evaluation/create_diagrams.py -i data/yago_evaluation_results.json -l model_name -o data/pr-diagram.png -ip
```
