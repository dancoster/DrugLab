#!/bin/bash

# Example code to run - bash pipeline/load_dataset.sh mimiciii 1.4

if [[ -z $1 && -z $2 ]]; then
    echo "Error: Add arguments..."
else
    echo "Start Download of " $2
    cd data/ && wget -r -N -c -np --user pavanreddy --ask-password https://physionet.org/files/$2/$4 --no-check-certificate && echo "Success"
fi