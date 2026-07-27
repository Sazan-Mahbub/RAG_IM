> [!WARNING]
> This codebase is under active development and does not yet fully reflect the [completed manuscript](https://arxiv.org/abs/2607.17508).

- This repository currently contains the implementation of our NeurIPS '24 workshop paper, [RAG-IM: Retrieval-Augmented Generation of Interpretable Models](https://openreview.net/forum?id=N4JhWiIUtg&referrer=%5Bthe%20profile%20of%20Sazan%20Mahbub%5D(%2Fprofile%3Fid%3D~Sazan_Mahbub1)), the non-archival workshop version of this work. 
- The completed preprint is currently available on arXiv as [Retrieval-Augmented Interpretable Learning: Towards Task-Specific Zero-Shot Models in Healthcare](https://arxiv.org/abs/2607.17508); the corresponding code will be released upon acceptance.

## Citation

If you use any part of this repository, please cite the completed manuscript:

```bibtex
@article{mahbub2026retrieval,
  title={Retrieval-Augmented Interpretable Learning: Towards Task-Specific Zero-Shot Models in Healthcare},
  author={Mahbub, Sazan and Ellington, Caleb and Li, Zhiyuan and Yang, Yixin and Kundu, Souvik and Lengerich, Ben and Xing, Eric P.},
  journal={arXiv preprint arXiv:2607.17508},
  year={2026}
}
```

or the earlier non-archival workshop version:

```bibtex
@inproceedings{mahbub2024ragim,
  title={From One to Zero: {RAG-IM} Adapts Language Models for Interpretable Zero-Shot Clinical Predictions},
  author={Mahbub, Sazan and Ellington, Caleb and Alinejad, Sina and Wen, Kevin and Luo, Yingtao and Lengerich, Ben and Xing, Eric P.},
  booktitle={NeurIPS 2024 Workshop on Adaptive Foundation Models: Evolving AI for Personalized and Efficient Learning},
  year={2024},
  url={https://openreview.net/forum?id=N4JhWiIUtg}
}
```

-------------------------------------------------

### Getting Started with the Workshop Version (Legacy):

#### Download raw data
Please download the MIMIC-IV dataset from the [official source](https://physionet.org/content/mimiciv/2.2/). The following files are required:
- HOSP.procedures_icd.csv
- HOSP.d_icd_procedures.csv
- HOSP.d_icd_diagnoses.csv
- HOSP.d_labitems.csv
- ED.diagnosis.csv
- labevents.csv

#### Run the experiments (training and testing)
> bash run_experiment.sh

#### Only test the already trained model
> bash run_test_only.sh

