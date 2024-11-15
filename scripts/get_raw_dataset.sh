#!/bin/bash

# Create the raw_data directory if it doesn't exist
mkdir -p raw_data
mkdir -p processed_data

# Download the files
wget -P raw_data/ https://ussf-ssac-23-soccer-gnn.s3.us-east-2.amazonaws.com/public/counterattack/women_imbalanced.pkl
wget -P raw_data/ https://ussf-ssac-23-soccer-gnn.s3.us-east-2.amazonaws.com/public/counterattack/men_imbalanced.pkl
wget -P raw_data/ https://ussf-ssac-23-soccer-gnn.s3.us-east-2.amazonaws.com/public/counterattack/women.pkl
wget -P raw_data/ https://ussf-ssac-23-soccer-gnn.s3.us-east-2.amazonaws.com/public/counterattack/men.pkl
wget -P raw_data/ https://ussf-ssac-23-soccer-gnn.s3.us-east-2.amazonaws.com/public/counterattack/combined.pkl