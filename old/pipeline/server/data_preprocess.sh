#!/bin/bash

# bash preprocess.sh '/home/gaga/data/physionet/mimiciii'


if [[ -z $1 ]]; then
    echo "Error: Add argument..."
else
    echo "Trial"
    python src/main.py --data $1 && echo "Success"
fi