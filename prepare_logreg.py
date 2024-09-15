from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import sklearn as k
import pandas as pd
import numpy as np
import pickle
import random
import torch
# import cudf
import json

import os
from tqdm import tqdm

from model import DecoupledLogisticRegression, create_log_reg_model

data_dir = "./data_v2"
home_dir = "./"
result_dir = "./data_v2"
model_dir = "./data_v2"

"""### Loading Dataset Pipeline"""

sorted_sub_hadm_df = pd.read_csv(f'{data_dir}/sub_hadm_ids_sorted.csv').values.squeeze()
sorted_sub_hadm = list(sorted_sub_hadm_df)
sub_hadm_id_to_ind = {(sub_id, hadm_id) : i for i, (sub_id, hadm_id) in enumerate(sorted_sub_hadm)}

tmp = list(sub_hadm_id_to_ind.items())
print(len(tmp))
print(f"{tmp[0][0]} : {tmp[0][1]}")

df_1stepNorm = pd.read_csv(f'{data_dir}/feature_matrix_1stepNorm.csv')
df_2stepNorm = pd.read_csv(f'{data_dir}/feature_matrix_2stepNorm.csv')
feature_matrix_1stepNorm = df_1stepNorm.values
feature_matrix_2stepNorm = df_2stepNorm.values
# feature_matrix_1stepNorm = torch.tensor(np_array_1stepNorm).half().cuda()
# feature_matrix_2stepNorm = torch.tensor(np_array_2stepNorm).half().cuda()

sorted_procedures_map = {}
# Opening JSON file
with open(f'{data_dir}/sorted_procedures_map.json', 'r') as openfile:

    # Reading from json file
    sorted_procedures_map = json.load(openfile)

sorted_procedures_list = list(sorted_procedures_map.items())
print(sorted_procedures_list[0])

len(sorted_procedures_list)

min_sample_size = 1 # NOTE: if we want, we can filter out tasks based on the minimum number of samples in them. By Default we are taking all the samples.
selected_procs = []
i = 0
while(i < len(sorted_procedures_list) and sorted_procedures_list[i][1]['count'] >= min_sample_size):
    proc = sorted_procedures_list[i][0]
    selected_procs.append(proc)
    i += 1

print(f"number of classes: {len(selected_procs)}")

from typing import Callable


def get_train_test_dataset(X, y, seed=42, test_size=0.5):
    data = pd.DataFrame(X)
    data['label'] = y

    positive = data[data['label'] == 1]
    negative = data[data['label'] == 0]

    n_positive = len(positive) // 2

    pos_train, pos_test = train_test_split(positive, test_size = test_size, random_state=seed)
    neg_train, neg_test = train_test_split(negative, test_size = test_size, random_state=seed)

    train_set = pd.concat([pos_train, neg_train])
    test_set = pd.concat([pos_test, neg_test])

    # Shuffle the final sets
    train_set = train_set.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_set = test_set.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Seperate features and labels
    X_train, y_train = train_set.drop('label', axis=1), train_set['label']
    X_test, y_test = test_set.drop('label', axis=1), test_set['label']

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

# random.seed(42)
# random.sample(list(range(10)), 3)
# random.seed(42000)
# random.sample([6,5,3,4,22,6,7,8,9,10], 3)

seed = 42 # for reproducibility
test_size = 0.5
X_tests_1, y_tests_1, X_trains_1, y_trains_1 =  [], [], [], []
X_tests_2, y_tests_2, X_trains_2, y_trains_2 =  [], [], [], []

decoupled_model_1StepNorm = DecoupledLogisticRegression(create_log_reg_model, n_classes = len(selected_procs))
decoupled_model_2StepNorm = DecoupledLogisticRegression(create_log_reg_model, n_classes = len(selected_procs))

for k, proc in tqdm(enumerate(selected_procs)):
    pos_sub_inds = sorted_procedures_map[proc]['sub_hadm_inds']
    all_inds = set(range(len(sorted_sub_hadm)))
    available_inds = list(all_inds - set(pos_sub_inds))
    random.seed(k * seed)
    neg_sub_inds = random.sample(available_inds, len(pos_sub_inds))

    pos_samples_2Step = feature_matrix_2stepNorm[pos_sub_inds]
    neg_samples_2Step = feature_matrix_2stepNorm[neg_sub_inds]
    pos_samples_1Step = feature_matrix_1stepNorm[pos_sub_inds]
    neg_samples_1Step = feature_matrix_1stepNorm[neg_sub_inds]

    X_k_2 = np.concatenate((pos_samples_2Step, neg_samples_2Step), axis=0)
    y_k_2 = np.concatenate((
        np.full(len(pos_samples_2Step), 1),
        np.full(len(neg_samples_2Step), 0)
    ))

    X_k_1 = np.concatenate((pos_samples_1Step, neg_samples_1Step), axis=0)
    y_k_1 = np.concatenate((
        np.full(len(pos_samples_2Step), 1),
        np.full(len(neg_samples_2Step), 0)
    ))

    try:
        X_train_k1, X_test_k1, y_train_k1, y_test_k1 = get_train_test_dataset(X_k_1, y_k_1, seed, test_size)
        X_train_k2, X_test_k2, y_train_k2, y_test_k2 = get_train_test_dataset(X_k_2, y_k_2, seed, test_size)
        
        decoupled_model_1StepNorm.fit(X_train_k1, y_train_k1)
        decoupled_model_2StepNorm.fit(X_train_k2, y_train_k2)
    except:
        X_train_k1 = X_k_1 * 0
        X_test_k1 = X_k_1
        y_train_k1 = y_k_1 * 0
        y_test_k1 = y_k_1

        X_train_k2 = X_k_2 * 0
        X_test_k2 = X_k_2
        y_train_k2 = y_k_2 * 0
        y_test_k2 = y_k_2

    X_tests_1.append(X_test_k1)
    y_tests_1.append(y_test_k1)
    X_trains_1.append(X_train_k1)
    y_trains_1.append(y_train_k1)
    X_tests_2.append(X_test_k2)
    y_tests_2.append(y_test_k2)
    X_trains_2.append(X_train_k2)
    y_trains_2.append(y_train_k2)

_preprocessed_data = [X_tests_1, y_tests_1, X_trains_1, y_trains_1, X_tests_2, y_tests_2, X_trains_2, y_trains_2]
pickle.dump(_preprocessed_data, open('all_preprocessed_data_ALL.pkl', 'wb'))

def get_test_train_accuracies_baselines(model, X_trains, X_tests, y_trains, y_tests):

    baseline1_test_accuracies = []
    baseline1_train_accuracies = []

    tot_test_accuracies = 0
    tot_train_accuracies = 0
    tot_f1 = 0

    n_test = 0
    n_train = 0

    for k, (X_train_k, X_test_k, y_train_k, y_test_k) in enumerate(zip(X_trains, X_tests, y_trains, y_tests)): # class to predict
        # Get test accuracies
        y_pred_test = model.predict(X_test_k, k)

        test_acc = accuracy_score(y_test_k, y_pred_test)

        baseline1_test_accuracies.append(test_acc)
        tot_test_accuracies += test_acc*len(y_test_k)
        n_test += len(y_test_k)

        f1 = f1_score(y_test_k, y_pred_test, average='macro')
        tot_f1 += f1*len(y_test_k)


        # Get train accuracies
        y_pred_train = model.predict(X_train_k, k)

        train_acc = accuracy_score(y_train_k, y_pred_train)
        baseline1_train_accuracies.append(train_acc)

        tot_train_accuracies += train_acc*len(y_train_k)
        n_train += len(y_train_k)


        print(f"Class {k} train accuracy: {train_acc}")
        print(f"Class {k} test accuracy: {test_acc}")
        print()

    overall_test_acc = tot_test_accuracies / n_test
    overall_f1 = tot_f1 / n_test
    overall_train_acc = tot_train_accuracies / n_train

    print(f"Overall train accuracy: {overall_train_acc}")
    print(f"Overall test accuracy: {overall_test_acc}")
    print(f"Overall test F1: {overall_f1}")

    return {"baseline1_test_accuracies": baseline1_test_accuracies,
            "baseline1_train_accuracies": baseline1_train_accuracies,
            "overall_test_acc": overall_test_acc,
            "overall_train_acc": overall_train_acc,
            "overall_f1": overall_f1}

result_1StepNorm = get_test_train_accuracies_baselines(decoupled_model_1StepNorm, X_trains_1, X_tests_1, y_trains_1, y_tests_1)

result_2StepNorm = get_test_train_accuracies_baselines(decoupled_model_2StepNorm, X_trains_2, X_tests_2, y_trains_2, y_tests_2)

import json

# the json file where the output must be stored
out_file = open(f"{result_dir}/result_LogReg_1StepNorm.json", "w")

json.dump(result_1StepNorm, out_file, indent = 6)

out_file.close()

# the json file where the output must be stored
out_file = open(f"{result_dir}/result_LogReg_2StepNorm.json", "w")

json.dump(result_2StepNorm, out_file, indent = 6)

out_file.close()

# Save model
with open(f'{model_dir}/decoupled_model_1StepNorm.pkl', 'wb') as f:
    pickle.dump(decoupled_model_1StepNorm, f)

with open(f'{model_dir}/decoupled_model_2StepNorm.pkl', 'wb') as f:
    pickle.dump(decoupled_model_2StepNorm, f)


"""
# Load model
with open(f'{model_dir}/decoupled_model_2StepNorm.pkl', 'rb') as f:
    decoupled_model_2StepNorm = pickle.load(f)

baseline1_test_accuracies, baseline1_train_accuracies, overall_test_acc, overall_train_acc, overall_f1 = get_test_train_accuracies_baselines(decoupled_model_2StepNorm)

# Load model
with open(f'{result_dir}/result_LogReg_1StepNorm.json', 'rb') as f:
    accs = json.load(f)

accs.keys()

plt.hist(accs['baseline1_test_accuracies'])
plt.show()
plt.clf()

plt.plot(accs['baseline1_test_accuracies'])
plt.show()
plt.clf()

plt.plot(accs['baseline1_test_accuracies'][:1000])
plt.show()
plt.clf()
"""