
import numpy as np
import pandas as pd
import torch
import sklearn as k
import gc
# import cudf
import json
import pickle
import matplotlib.pyplot as plt
# from contextualized.easy import ContextualizedRegressor, ContextualizedClassifier
# from transformers import DistilBertModel, DistilBertTokenizer, DistilBertConfig
import os
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import gc
import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
# from transformers import PreTrainedTokenizerFast, BertTokenizerFast, BertModel
# from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


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


####
class CLLMdataset(Dataset):
    def __init__(self, split, X, Y, context_encoding, logreg_models, nbr_indices, context_idx=None):
        assert split in {'train', 'test', 'val'}
        if split == 'train':
            assert context_idx is None
        else:
            assert context_idx is not None
            self.context_idx = context_idx

        self.split = split

        self.X = X
        self.Y = Y
        self.context_encoding = context_encoding[:len(X)].astype(np.float32)
        self.logreg_models = logreg_models.astype(np.float32)
        self.nbr_indices = nbr_indices
        
        if split != 'train':
            x_temp = self.X[self.context_idx]
            x = np.concatenate([np.ones([x_temp.shape[0], 1]), x_temp], 1).astype(np.float32)
            y_pred_baseline = (x @ self.logreg_models[self.context_idx] > 0.0).astype(int)
            self.baseline_accuracies = (y_pred_baseline == Y[self.context_idx]).mean()
            # print(f'Baseline accuracy for task {self.context_idx} is {self.baseline_accuracies}')

        
        assert len(self.X) == self.context_encoding.shape[0]
        assert len(self.Y) == self.context_encoding.shape[0]

        self.REPEATE_COUNT = 10

        

    def __len__(self):
        if self.split == 'train':
            return self.context_encoding.shape[0] * self.REPEATE_COUNT  ## NOTE: iterate over the tasks/contexts during training.
        else:
            if self.X[self.context_idx].shape[0] == 2:
                return 2
            return self.X[self.context_idx].shape[0]  ## NOTE: iterate over the samples within the task/context during testing/validation.

    def get_one_sample(self, context_idx, sample_idx):
        c = self.context_encoding[context_idx]
        x = np.concatenate([np.ones((1,)), self.X[context_idx][sample_idx]]).astype(np.float32)
        y = self.Y[context_idx][sample_idx].astype(np.float32)
        
        ## switch based on dataset size
        # if self.X[context_idx].shape[0] >= 50:
        #     self_model = self.logreg_models[context_idx]
        # else:
        #     self_model = np.zeros_like(self.logreg_models[0])
        try:
            self_model = self.logreg_models[context_idx]
            if self.X[context_idx].shape[0] < 50:   ## NOTE: switching-off the residual/skip connection
                self_model0 = self_model * 0
            else: 
                self_model0 = self_model.copy()
        except:
            self_model = np.zeros_like(self.logreg_models[0])
            self_model0 = self_model.copy()
        
        nbr_models = self.logreg_models[self.nbr_indices[context_idx]]
        nbr_embs = self.context_encoding[self.nbr_indices[context_idx]]
        # print(int(self_model @ x >  0), y)
        return c, x, y, self_model0, self_model, nbr_models, nbr_embs
    
    def __getitem__(self, idx):
        if self.split == 'train':
            context_idx = idx % self.context_encoding.shape[0]
            sample_idx  = np.random.randint(low=0, high=self.X[context_idx].shape[0])
        else:
            context_idx = self.context_idx
            sample_idx  = idx
        return self.get_one_sample(context_idx, sample_idx)


