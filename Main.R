system("mkdir -p processed_data")
system("mkdir -p raw_data")
system("mkdir -p models")
system("conda env create -f environment.yml")

source("ObtainData.R")
source("PrepData.R")
source("Analysis.R")

