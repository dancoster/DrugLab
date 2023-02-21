# Medication-Labtests
Repository for `Medication <> Lab Test interaction discovery analysis`.

## Directory Structure

    .
    ├── data                    # Datasets - Not pushed (Drive link in [dataset section](##Dataset))
    ├── docs                    # Documentation files (Will update shortly)
    ├── old                    # Previous version
    |    ├── src                     # Source Code 
    |    │   ├── preprocessing               # Data Preprocessing
    |    |   |   ├── preprocess.py             # Data Loading and preprocessing class
    |    |   |   └── stratify.py               # Stratification class
    |    │   ├── analysis                    # Data Analysis
    |    |   |   ├── analysis.py               # Analysis Parent Class
    |    |   |   ├── inputevents_analysis.py   # Inputevents table - Before and After, regression and trend analysis
    |    |   |   ├── presc_analysis.py         # Prescriptions table - Before and After, regression and trend analysis
    |    |   |   └── time_effect_analysis.py   # Time effct analysis
    |    │   ├── modeling                    # Data Modeling
    |    │   └── visualize                   # Visualization of data analysis
    |    |       └── time_effect_visualize.py   # Time effct Visualization
    |    ├── notebooks                     # Notebook files
    |    │   ├── Dataset                     # Datset access
    |    │   ├── Drugs                       # Drug data taken from chartevents table
    |    │   ├── Medication                  # Medication data from inputevents_mv (Metavision) table
    |    │   ├── Medication-Regression       # Medication data from inputevents_mv (Metavision) table + Regression done to perform analysis
    |    │   ├── Prescription                # Prescription data from prescription table
    |    │   ├── Others                      # MIMIC Extract and Other analysis
    |    |   └── sql_queries                 # SQL Queries used to read from MIMIC-III Postgre SQL
    |    ├── pipeline                # Automation pipelines and scripts Database
    |    └── plots                   # Plots for data analysis and visualization (Older plots on drive - check [plots section](##Plots))
    ├── mimic-notebook-analysis                     # MIMIC Notebook files
    ├── hirid-notebook-analysis                     # HIRID Notebook files
    ├── src                     # Source Code 
    │   ├── parsers               # Dataset parsers
    |   |   ├── parser.py             # Parser interface
    |   |   ├── mimic.py              # MIMIC Dataset Parser
    |   |   └── hirid.py              # MIMIC Dataset Parser
    │   ├── modling               # Data Analysis and modeling
    |   |   ├── querier.py            # dataset querier - lab test <> medication pair generation code
    |   |   ├── discovery.py          # Before and After and trend analysis
    |   |   └── plots.py              # Time effct analysis
    │   └── utils                 # Utils functions and classes
    |       ├── utils.py              # utils
    |       └── constants.py          # Constant values - lab id mappings for all datasets
    ├── results                 # Results 
    └── README.md

## Dataset
Drive Link to Datasets : https://drive.google.com/drive/folders/1-ARL03M0t8yGB3xiy_k1lhmQBHJWToFK?usp=share_link

    .
    ├── mimic_extract                                            # MIMIC Extract Dataset
    ├── mimiciii                                                 # MIMIC-III Dataset
    ├── hirid-a-high-time-resolution-icu-dataset-1.1.1           # HIRID Dataset 
    └── itemid_to_variable_map.csv                               # MIMIC Extract - itemid to lab test/medication mapping

## Results
`Needs to be updated`

## Plots
`Needs to be updated`

## Running Code
1. Setup VM (With the repo as root directory). Recommend using Python 3.9.6
```
python3 -m venv venv
pip3 install -r requirements.txt
```
2. Use either `main.py` or `main.ipynb` to run the code

