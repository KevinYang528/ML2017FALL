#!/bin/bash
wget -O "my_model_0.66453.h5" "https://github.com/KevinYang528/ML2017FALL/releases/download/0.0.0/my_model_0.66453.h5"
wget -O "my_model_0.66620.h5" "https://github.com/KevinYang528/ML2017FALL/releases/download/0.0.0/my_model_0.66620.h5"
wget -O "my_model_0.67093.h5" "https://github.com/KevinYang528/ML2017FALL/releases/download/0.0.0/my_model_0.67093.h5"
wget -O "my_model_0.67679.h5" "https://github.com/KevinYang528/ML2017FALL/releases/download/0.0.0/my_model_0.67679.h5"
wget -O "my_model_0.67846.h5" "https://github.com/KevinYang528/ML2017FALL/releases/download/0.0.0/my_model_0.67846.h5"
python3 test.py $1 $2