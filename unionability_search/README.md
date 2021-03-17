# Unionability Search Evaluation

Here you can find tools for an evaluation of Web table embeddings and word embeddings for the calculation of unionability scores.
Those unionability sores could then be used for implementing a [table union search application](https://dl.acm.org/doi/pdf/10.14778/3192965.3192973).

### Setup

For the evaluation we used the [Table Union Search Benchmark](https://github.com/RJMillerLab/table-union-search-benchmark) constructed from open data tables.
After downloading the benchmark, you can run the evaluation for different embedding  models using the by executing the `calculate_unionability.py` script:

```
python3 unionability_search/calculate_unionability.py -e path/to/embedding_model -et web-table -b path/to/unionability_benchmark/ -s 1000 -o data/unionablity_eval_results.json
```

The results can be plotted using `create_diagram.py`:

```
python3 unionability_search/create_diagrams.py -i data/unionablity_eval_results.json -l model_name -o data/pr-unionability.png
```
