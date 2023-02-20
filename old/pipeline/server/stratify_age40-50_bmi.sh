#!/bin/bash

python3 src/main.py -i --window 1 24 --meds 200 --between-meds 1 2 --sub-pairs 50 --stratify -a 40 50 -e WHITE -b 18.5 24.9 -g M
python3 src/main.py -i --window 1 24 --meds 200 --between-meds 1 2 --sub-pairs 50 --stratify -a 40 50 -e WHITE -b 18.5 24.9 -g F
python3 src/main.py -i --window 1 24 --meds 200 --between-meds 1 2 --sub-pairs 50 --stratify -a 40 50 -e WHITE -b 25 30 -g M
python3 src/main.py -i --window 1 24 --meds 200 --between-meds 1 2 --sub-pairs 50 --stratify -a 40 50 -e WHITE -b 25 30 -g F