#!/bin/bash
wget -O "model_0.66453.h5" "https://www.dropbox.com/s/ugt8xqoaptjj7cg/model_0.66453.h5?dl=1"
wget -O "model_0.66620.h5" "https://www.dropbox.com/s/fuzg0ykndk7zokj/model_0.66620.h5?dl=1"
wget -O "model_0.67093.h5" "https://www.dropbox.com/s/gcjng9byqawjyau/model_0.67093.h5?dl=1"
wget -O "model_0.67679.h5" "https://www.dropbox.com/s/48i7l3844pqz5ge/model_0.67679.h5?dl=1"
wget -O "model_0.67846.h5" "https://www.dropbox.com/s/krwpmomqmop1xtj/model_0.67846.h5?dl=1"
python3 test.py $1 $2