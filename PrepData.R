system("python scripts/convert_data.py")
system("pytest ./test/")
system("python scripts/clean_data.py")
system("pytest ./test/")
system("python scripts/label_data.py processed_data/men_imbalanced_node_features.parquet")
# manual check code
system("python scripts/preprocess_data.py")