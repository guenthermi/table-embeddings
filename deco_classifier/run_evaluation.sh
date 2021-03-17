python3 evaluate_classifier.py -i data/features-deco100-simple.pkl -t web-table -e ../web_table_embeddings/data/web_table_embeddings_combo150.bin -o data/combo -f embeddings deco -it 50 -c gnn random-forest
python3 evaluate_classifier.py -i data/features-deco100-simple.pkl -t web-table -e ../web_table_embeddings/data/baseline_model150.bin -o data/base -f embeddings deco -it 50 -c gnn random-forest
python3 evaluate_classifier.py -i data/features-deco100-simple.pkl -t web-table -e ../web_table_embeddings/data/web_table_embeddings_plain150.bin -o data/plain -f embeddings deco -it 50 -c gnn random-forest
python3 evaluate_classifier.py -i data/features-deco100-simple.pkl -t fasttext -e data/wiki.en.bin -o data/fasttext -f embeddings deco -it 50 -c gnn random-forest
