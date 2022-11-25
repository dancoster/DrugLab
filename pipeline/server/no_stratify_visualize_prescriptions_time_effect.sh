#! /bin/bash

python3 src/main.py -v --window 1 48 --table prescriptions --visualize "Iso-Osmotic Dextrose" Hematocrit
python3 src/main.py -v --window 1 48 --table prescriptions --visualize NS Hematocrit
python3 src/main.py -v --window 1 48 --table prescriptions --visualize "Magnesium Sulfate" Hematocrit
python3 src/main.py -v --window 1 48 --table prescriptions --visualize "Iso-Osmotic Dextrose" "Red Blood Cells"
python3 src/main.py -v --window 1 48 --table prescriptions --visualize "Magnesium Sulfate" "Red Blood Cells"
python3 src/main.py -v --window 1 48 --table prescriptions --visualize NS "Red Blood Cells"
python3 src/main.py -v --window 1 48 --table prescriptions --visualize SW "Red Blood Cells"
python3 src/main.py -v --window 1 48 --table prescriptions --visualize Insulin Hematocrit
python3 src/main.py -v --window 1 48 --table prescriptions --visualize Insulin "Red Blood Cells"
python3 src/main.py -v --window 1 48 --table prescriptions --visualize Insulin Glucose