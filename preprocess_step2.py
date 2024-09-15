from transformers import DistilBertModel, DistilBertTokenizer, DistilBertConfig
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import torch
import sklearn as k
# import cudf
import json
import pickle
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

data_dir = "./data_v2"
home_dir = "./"
write_data_dir = "./data_v2"


def preprocess_labevents():
    selected_columns = ['itemid', 'subject_id', 'hadm_id', 'charttime', 'valuenum', 'ref_range_lower', 'ref_range_upper']
    subject_labevents = pd.read_csv(f'{data_dir}/filtered_labevents__all_icd_code.csv', usecols=selected_columns)
    subject_labevents = subject_labevents.dropna(subset=["hadm_id"])
    subject_labevents = subject_labevents.dropna(subset=["valuenum", "ref_range_lower", "ref_range_upper"])
    subject_labevents = subject_labevents.groupby(['itemid', 'subject_id', 'hadm_id']).agg({
      'valuenum': "mean",
      "ref_range_upper": "first",
      "ref_range_lower": "first"
    }).reset_index()
    return subject_labevents

# Function to normalize valuenum using ref_range_upper and ref_range_lower
def normalize_value_1step(row):
    range = (row['ref_range_upper'] - row['ref_range_lower'])

    if range == 0:
        return None
    return np.round((row['valuenum'] - row['ref_range_lower']) / range, 6)

def normalize_value_2step(row):
    norm_val = row['normalized_valuenum_1step']
    output = norm_val
    if norm_val >= 0 and norm_val <= 1:
      output = 0
    if norm_val > 1:
      output = norm_val - 1
    return output

def make_save_feature_matrices(subject_labevents, sub_hadm_id_to_ind, item_ids_to_ind):
    feature_matrix_1stepNorm = torch.full([len(unique_combinations), len(item_ids)], 0.5) # should be filled with 0.5
    feature_matrix_2stepNorm = torch.zeros([len(unique_combinations), len(item_ids)]) # should be filled with 0
    for _, labevent in subject_labevents.iterrows():
      sub_id  = labevent['subject_id']
      hadm_id = labevent['hadm_id']
      sub_hadm_ind = sub_hadm_id_to_ind[(sub_id, hadm_id)]
      item_id = labevent['itemid']
      item_ind = item_ids_to_ind[item_id]
      one_step_norm = labevent['normalized_valuenum_1step']
      two_step_norm = labevent['normalized_valuenum_2step']
      feature_matrix_1stepNorm[sub_hadm_ind, item_ind] = one_step_norm
      feature_matrix_2stepNorm[sub_hadm_ind, item_ind] = two_step_norm

    df_1stepNorm = pd.DataFrame(feature_matrix_1stepNorm)
    df_2stepNorm = pd.DataFrame(feature_matrix_2stepNorm)
    df_1stepNorm.to_csv(f'{write_data_dir}/feature_matrix_1stepNorm.csv', index=False)
    df_2stepNorm.to_csv(f'{write_data_dir}/feature_matrix_2stepNorm.csv', index=False)
    return feature_matrix_1stepNorm, feature_matrix_2stepNorm

def convert_subject_id_to_ind(row):
  output = []
  for sub_hadm_tup in list(row['sub_hadm_tuples']):
    if sub_hadm_tup in sub_hadm_id_to_ind:
      sub_hadm_ind = sub_hadm_id_to_ind[sub_hadm_tup]
      output.append(sub_hadm_ind)
  if len(output) == 0:
    return None
  return output

def add_definition(row):
  return proc_def[row['icd_code']]

def make_proc_to_sub_hadm(subject_proc):
  subject_hadm_proc = subject_proc.groupby(['subject_id', 'hadm_id', 'icd_code']).agg(
    count = ('icd_version', 'size')
  ).reset_index()
  
  proc_to_sub_hadm = subject_hadm_proc.groupby('icd_code').apply(
    lambda row: list(set(zip(row['subject_id'], row['hadm_id'])))
  ).reset_index()

  proc_to_sub_hadm.columns = ['icd_code', 'sub_hadm_tuples']
  proc_to_sub_hadm['definition'] = proc_to_sub_hadm.apply(add_definition, axis=1)
  proc_to_sub_hadm['sub_hadm_inds'] = proc_to_sub_hadm.apply(convert_subject_id_to_ind, axis=1)
  proc_to_sub_hadm = proc_to_sub_hadm.dropna(subset=["sub_hadm_inds"])
  proc_to_sub_hadm['count'] = proc_to_sub_hadm.apply(lambda row: len(row['sub_hadm_inds']), axis=1)
  proc_to_sub_hadm = proc_to_sub_hadm.sort_values(by=['count'], ascending=False)
  return proc_to_sub_hadm

def make_save_sorted_procedures_map(proc_to_sub_hadm):
    sorted_procedures_map = {}
    for row in proc_to_sub_hadm.itertuples(index=False):
      icd_code, sub_hadm_tuples, definition, sub_hadm_inds, count = row
      sorted_procedures_map[icd_code] = {
          'definition': definition,
          'count': count,
          "sub_hadm_inds": list(sub_hadm_inds)
          # "subject_inds": torch.tensor(list(subject_inds), dtype=torch.long)
      }
    # Serializing json
    json_object = json.dumps(sorted_procedures_map, indent=4)
    # Writing to sample.json
    with open(f"{write_data_dir}/sorted_procedures_map.json", "w") as outfile:
        outfile.write(json_object)
    
    return sorted_procedures_map

def get_bert_class_embeddings(class_description, max_length=256):
    """
    Args:
        class_description (list): list of class descriptions
    Returns:
        nd.array: (n_classes, bert_embedding_size=764)
    """
    # Get embeddings for every class description
    batch_class_encoding = tokenizer.batch_encode_plus(
        class_description,
        max_length=max_length,
        truncation=True,
        pad_to_max_length=True,
        return_attention_mask=True,
        padding='max_length',
        return_tensors='pt',
    )



    with torch.no_grad():
        model.eval()
        batch_class_encoding = {k: v.to(device) for k, v in batch_class_encoding.items()}

        outputs = model(**batch_class_encoding)
        embeddings = outputs.last_hidden_state # (batch_size, sequence length, embedding_size)

        # Average token embeddings per item in batch
        class_embeddings = embeddings.mean(dim=1) # (batch_size, embedding_size)

    # print(class_embeddings.shape)
    return class_embeddings.half().cpu().numpy()


if __name__ == "__main__":
  subject_labevents = preprocess_labevents()
  # Get all unique combinations of the two columns
  unique_combinations = subject_labevents[["subject_id", "hadm_id"]].drop_duplicates()
  unique_combinations = list(zip(unique_combinations["subject_id"], unique_combinations["hadm_id"]))
  sorted_sub_hadm = sorted(unique_combinations)
  sub_hadm_id_to_ind = {(sub_id, hadm_id) : i for i, (sub_id, hadm_id) in enumerate(sorted_sub_hadm)}
  sorted_sub_hadm_df = pd.DataFrame(sorted_sub_hadm)
  sorted_sub_hadm_df.to_csv(f'{write_data_dir}/sub_hadm_ids_sorted.csv', index=False)


  subject_labevents['normalized_valuenum_1step'] = subject_labevents.apply(normalize_value_1step, axis=1)
  subject_labevents = subject_labevents.dropna(subset=["normalized_valuenum_1step"])
  subject_labevents['normalized_valuenum_2step'] = subject_labevents.apply(normalize_value_2step, axis=1)
  item_ids = sorted(subject_labevents['itemid'].unique())
  item_ids_to_ind = {item_id : i for i, item_id in enumerate(item_ids)}
  item_ids_df = pd.DataFrame(item_ids)
  item_ids_df.to_csv(f'{write_data_dir}/item_ids_sorted.csv', index=False)
  feature_matrix_1stepNorm, feature_matrix_2stepNorm = make_save_feature_matrices(subject_labevents, sub_hadm_id_to_ind, item_ids_to_ind)

  subject_proc = pd.read_csv(f"{data_dir}/HOSP.procedures_icd.csv")
  subject_proc = subject_proc[subject_proc["icd_version"] == 10]
  proc_def_df = pd.read_csv(f"{data_dir}/HOSP.d_icd_procedures.csv")
  proc_def = {}

  for row in proc_def_df.itertuples(index=False):
    icd_code, icd_vers, descrip = row

    if icd_vers == 10:
      proc_def[icd_code] = descrip

  proc_to_sub_hadm = make_proc_to_sub_hadm(subject_proc)
  sorted_procedures_map = make_save_sorted_procedures_map(proc_to_sub_hadm)
  sorted_procedures_list = list(sorted_procedures_map.items())
  all_procs_def = [item[1]['definition'] for item in sorted_procedures_list]


  device = torch.device('cpu')
  if torch.cuda.is_available():
    device = torch.device('cuda:0')
  model_name = "distilbert-base-uncased"
  tokenizer = DistilBertTokenizer.from_pretrained(model_name)
  model = DistilBertModel.from_pretrained(model_name)

  model = model.to(device)

  bert_embeddings = [get_bert_class_embeddings(x[0]) for x in all_procs_def]
  bert_embeddings = np.array(bert_embeddings).astype(np.float32)
  bert_class_embeddings_df = pd.DataFrame(bert_embeddings)
  bert_class_embeddings_df.to_csv(f'{write_data_dir}/bert_class_embeddings.csv', index=False)
