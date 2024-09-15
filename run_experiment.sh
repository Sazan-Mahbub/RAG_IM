python preprocess_step1.py &> logs/preprocess_step1.log
python preprocess_step2.py &> logs/preprocess_step2.log
python prepare_logreg.py &> logs/prepare_logreg.log
python train.py &> logs/train.log
python inference_all.py &> logs/inference_all.log
