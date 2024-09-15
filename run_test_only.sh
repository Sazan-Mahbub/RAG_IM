python preprocess.py &> logs/preprocess.log
python prepare_logreg.py &> logs/prepare_logreg.log
python inference_all.py &> logs/inference_all.log
