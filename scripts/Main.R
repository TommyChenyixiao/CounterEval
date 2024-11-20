# For now, it is just used to list the order of the scripts to be run
system("sh scripts/get_raw_dataset.sh")
system("python scripts/convert_data.py")
system("pytest ./test/")
system("python scripts/clean_data.py")
system("pytest ./test/")
system("python scripts/label_data.py processed_data/men_imbalanced_node_features.parquet")
# system("python scripts/label_data.py processed_data/women_imbalanced_node_features.parquet")