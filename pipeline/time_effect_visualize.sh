#! /bin/bash

# python3 src/main.py -v --window 1 48 --table inputevents --visualize "Packed Red Blood Cells"	Hematocrit
# python3 src/main.py -v --window 1 48 --table inputevents --visualize "Packed Red Blood Cells" Hemoglobin
# python3 src/main.py -v --window 1 48 --table inputevents --visualize "Packed Red Blood Cells" "Red Blood Cells"
# python3 src/main.py -v --window 1 48 --table inputevents --visualize "OR Crystalloid Intake" pO2
# python3 src/main.py -v --window 1 48 --table inputevents --visualize "OR Cell Saver Intake" pO2

python3 src/main.py -v -p --window 1 48 --table prescriptions --visualize Insulin Glucose --meds 9400 --sub-pairs 1000

# python3 src/main.py -v --window 1 48 --table inputevents --visualize "Insulin - Regular" Glucose