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

import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader

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

from typing import Callable



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



def one_epoch(dataloader, split, model, device, optimizer=None):
    assert split in {'train', 'test', 'val'}
    y_true = []
    y_pred = []
    y_pred_baseline = []
    losses = []
    if split == 'train':
        model.train()
    else:
        model.eval()
        # print('dataloader.dataset.context_idx:', dataloader.dataset.context_idx)
        # print('dataloader.dataset.baseline_accuracies:', dataloader.dataset.baseline_accuracies)
    

    if split == 'train':
        _enumer_obj = tqdm(enumerate(dataloader))
    else:
        _enumer_obj = enumerate(dataloader)
        
    for b_idx, (c, x, y, self_model0, self_model, nbr_models, nbr_embs) in _enumer_obj:
        # print(b_idx, c.shape, x.shape, y.shape, self_model.shape, nbr_models.shape, nbr_embs.shape)
        
        c = c.to(device)
        x = x.to(device)
        y = y.to(device)
        self_model0 = self_model0.to(device)
        self_model = self_model.to(device)
        nbr_models = nbr_models.to(device)
        nbr_embs = nbr_embs.to(device)

        pred_baseline = (torch.diag(x @ self_model.T) > 0.0).float()
        # baseline_accuracies = (y_pred_baseline == y).float().mean()
        # print(pred_baseline == y, y_pred_baseline, y)
        # print(f'Baseline accuracy for batch {b_idx} is {baseline_accuracies}')
        # print(x.shape, self_model.shape)
        # pred_baseline = (torch.bmm(x.unsqueeze(1), self_model.unsqueeze(2)).view(-1) > 0).float()
        # print((pred_baseline == y).float().mean()); raise
        
        model.zero_grad()
        
        if split == 'train':
            pred, loss = model(c, x, y, self_model0, nbr_models, nbr_embs)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                pred, loss = model(c, x, y, self_model0, nbr_models, nbr_embs)
                pred = pred.detach()
                loss = loss.detach()
        y_true += y.detach().cpu().numpy().tolist()
        y_pred += pred.detach().cpu().numpy().tolist()
        y_pred_baseline += pred_baseline.detach().cpu().numpy().tolist()
        losses += [loss.cpu().item()]
        

    y_true = np.array(y_true).astype(int)
    y_pred = (np.array(y_pred) > 0.5).astype(int)
    y_pred_baseline = np.array(y_pred_baseline).astype(int)
    losses = np.array(losses)
    
    ## metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    acc_b = accuracy_score(y_true, y_pred_baseline)
    precision_b, recall_b, f1_b, support_b = precision_recall_fscore_support(y_true, y_pred_baseline, zero_division=0)

    if split == 'train':
        print('Split:', split)
        print("\taccuracy: ", acc, f'(baseline: {acc_b})')
        print("\tprecision: ", precision, f'(baseline: {precision_b})')
        print("\trecall: ", recall, f'(baseline: {recall_b})')
        print("\tf1: ", f1, f'(baseline: {f1_b})')
        print("\tsupport: ", support, f'(baseline: {support_b})')
        print("\tAvg loss:", losses.mean())
        print()

    return y_true, y_pred, losses, acc, precision, recall, f1, support, acc_b, precision_b, recall_b, f1_b, support_b

