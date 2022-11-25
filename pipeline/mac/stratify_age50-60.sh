#!/bin/bash

python src/main.py -i --window 1 24 --meds 200 --between-meds 1 2 --sub-pairs 50 --stratify -a 50 60 -e WHITE -b 1 100 -g M --data "/Volumes/GoogleDrive/My Drive/TAU/Code/DrugLab/data"
python src/main.py -i --window 1 24 --meds 200 --between-meds 1 2 --sub-pairs 50 --stratify -a 50 60 -e WHITE -b 1 100 -g F --data "/Volumes/GoogleDrive/My Drive/TAU/Code/DrugLab/data"