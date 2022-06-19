# Medication-Labtests
Repository for medication<>lab-test pair.

`Note: Still Updating the README file`

## Directory Structure

    .
    ├── docs                    # Documentation files (Will update shortly)
    ├── sql_queries             # SQL Queries used to read from MIMIC-III Postgre SQL Database
    ├── src                     # Source files (Notebooks with code)
    │   ├── Dataset                     # Datset access
    │   ├── Drugs                       # Drug data taken from chartevents table
    │   ├── Medication                  # Medication data from inputevents_mv (Metavision) table
    │   ├── Medication-Regression       # Medication data from inputevents_mv (Metavision) table + Regression done to perform analysis
    │   ├── Prescription                # Prescription data from prescription table
    │   └── Others                      # MIMIC Extract and Other analysis
    └── README.md

## Dataset
Drive Link to Datasets : https://drive.google.com/drive/folders/1KGo8IhGQDtQEUW5tfUzqQMpsvy3yE9uf?usp=sharing

    .
    ├── eicu-crd                    # eICU-CRD Dataset
    ├── mimiciii                    # MIMIC-III Dataset
    └── mimiciv                     # MIMIC-IV Dataset
    

## Results
Drive Link to CSV Files : https://drive.google.com/drive/folders/1GVhYml7dnjQpYCfJFtatNJ2M9R03uinF?usp=sharing

    .
    ├── test                  # 
    │   ├── File1               # Load and stress tests
    │   ├── File2               # End-to-end, integration tests (alternatively `e2e`)
    │   └── File3               # Unit tests
    └── test                  # 
    │   ├── File1               # Load and stress tests
    │   ├── File2               # End-to-end, integration tests (alternatively `e2e`)
    │   └── File3               # Unit tests
    └── test                  # 
    │   ├── File1               # Load and stress tests
    │   ├── File2               # End-to-end, integration tests (alternatively `e2e`)
    │   └── File3               # Unit tests
    └── test                  # 
    │   ├── File1               # Load and stress tests
    │   ├── File2               # End-to-end, integration tests (alternatively `e2e`)
    │   └── File3               # Unit tests
    └── test                  # 
        ├── benchmarks          # Load and stress tests
        ├── integration         # End-to-end, integration tests (alternatively `e2e`)
        └── unit

## Plots
Drive Link to Medication<>Labtest Pair value plots : https://drive.google.com/drive/folders/1HV8gQ5LA3HhJc4ACL0DbOFec7bkweyk5?usp=sharing
