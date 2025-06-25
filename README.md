# RAG-IM
Implementation of RAG-IM (Retrieval Augmented Generation of Interpretable Models). 

**CAUTION:** The codebase is actively being modified.

## Download raw data
Please download the MIMIC-IV dataset from the [official source](https://physionet.org/content/mimiciv/2.2/). The following files are required:
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

