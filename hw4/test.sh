#!/bin/bash
wget -O "w2v/word2vec_mark.pkl" "https://www.dropbox.com/s/e8v9gntbfoeys1t/word2vec_mark.pkl?dl=1"
wget -O "model/model-029-0.83725.h5" "https://www.dropbox.com/s/yoqjezkav2dko5j/model-029-0.83725.h5?dl=1"
python3 test.py $1 $2