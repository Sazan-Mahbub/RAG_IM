# RAG_IM
This repository contains the official implementation of RAG-IM (Retrieval Augmented Generation of Interpretable Models). 

## Download raw data
Please download the original dataset from [here](https://physionet.org/content/mimiciv/3.0/). The following files are required:
- HOSP.procedures_icd.csv
- HOSP.d_icd_procedures.csv
- HOSP.d_icd_diagnoses.csv
- HOSP.d_labitems.csv
- ED.diagnosis.csv
- labevents.csv

## Run the experiments (training and testing)
> bash run_experiment.sh

## Only test the already trained model
> bash run_test_only.sh

