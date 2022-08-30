#!/bin/bash

python3 src/main.py -v --visualize "Insulin - Regular" Glucose --table inputevents --window 1 24 --before-window 2 20 --after-window 1 10 --meds 200 --sub-pairs 50 --stratify -a 50 80 -e WHITE -b 25 30 -g M