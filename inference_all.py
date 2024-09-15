from utils import *
from model import *
from datamodule import *


import os
from tqdm import tqdm
data_dir = "./data_v2"
home_dir = "./"
result_dir = "./results"

sorted_sub_hadm_df = pd.read_csv(f'{data_dir}/sub_hadm_ids_sorted.csv').values.squeeze()
sorted_sub_hadm = list(sorted_sub_hadm_df)
sub_hadm_id_to_ind = {(sub_id, hadm_id) : i for i, (sub_id, hadm_id) in enumerate(sorted_sub_hadm)}

df_1stepNorm = pd.read_csv(f'{data_dir}/feature_matrix_1stepNorm.csv')
df_2stepNorm = pd.read_csv(f'{data_dir}/feature_matrix_2stepNorm.csv')
feature_matrix_1stepNorm = df_1stepNorm.values
feature_matrix_2stepNorm = df_2stepNorm.values

logRegModels = pickle.load(open(f'{data_dir}/decoupled_model_1StepNorm.pkl', 'rb'))

logRegModel_weights = []
for i in range(len(logRegModels.models_binary_decoupled)):
    coef = logRegModels.models_binary_decoupled[i].coef_[0]
    intercept = logRegModels.models_binary_decoupled[i].intercept_
    logRegModel_weights += [np.concatenate([intercept, coef])]
logRegModel_weights = np.array(logRegModel_weights)
# print(logRegModel_weights.shape)
# exit()


sorted_procedures_map = {}
# Opening JSON file
with open(f'{data_dir}/sorted_procedures_map.json', 'r') as openfile:
    # Reading from json file
    sorted_procedures_map = json.load(openfile)


# In[12]:


sorted_procedures_list = list(sorted_procedures_map.items())
# print(sorted_procedures_list[0])


# In[13]:


len(sorted_procedures_list)


# In[14]:


min_sample_size = 1
selected_procs = []
i = 0
while(i < len(sorted_procedures_list) and sorted_procedures_list[i][1]['count'] >= min_sample_size):
  proc = sorted_procedures_list[i][0]
  selected_procs.append(proc)
  i += 1


# In[15]:


print(f"number of classes: {len(selected_procs)}")


# In[26]:


X_tests_1, y_tests_1, X_trains_1, y_trains_1, X_tests_2, y_tests_2, X_trains_2, y_trains_2 = pickle.load(open(f'{data_dir}/all_preprocessed_data_ALL.pkl', 'rb'))

bert_class_embeddings_df = pd.read_csv(f'{data_dir}/bert_class_embeddings.csv')
bert_class_embeddings = bert_class_embeddings_df.values[:len(X_tests_1)]
assert bert_class_embeddings.shape[0] == len(X_tests_1)


# In[133]:
##### find top K neighbors
K = 10
cos_sim_matrix = cosine_similarity(bert_class_embeddings, bert_class_embeddings) # Calculate cosine similarity
np.fill_diagonal(cos_sim_matrix, -np.inf)
cos_sim_matrix = cos_sim_matrix[:, :4412]
nbr_indices = np.argsort(cos_sim_matrix, axis=1)[:, -K:] # Find indices of top K closest vectors for each embedding

####### dataloaders
batch_size = 1024

test_set = CLLMdataset(split='test', X=X_tests_1, Y=y_tests_1, 
                                  context_encoding=bert_class_embeddings, 
                                  logreg_models=logRegModel_weights,
                                  nbr_indices=nbr_indices,
                                  context_idx=0)
test_loader = DataLoader(test_set, batch_size=batch_size*4, shuffle=False, num_workers=1)


def eval_all(start, end, verbose=0):
    # test_sets = []
    # test_loaders = []
    precision_all = []
    recall_all = []
    f1_all = []
    acc_all = []

    precision_all_baseline = []
    recall_all_baseline = []
    f1_all_baseline = []
    acc_baseline = []
    ttl_cnt = []
    for i in range(start, end):
        if verbose in {1,2}:
            print('Task:', i)
        test_loader.dataset.context_idx = i
        x = np.concatenate([np.ones([X_tests_1[i].shape[0], 1]), X_tests_1[i]], 1).astype(np.float32)
        # y_pred_baseline = (x @ logRegModel_weights[i] > 0.0).astype(int)
        
        y_true, y_pred, losses, acc, precision, recall, f1, support, acc_b, precision_b, recall_b, f1_b, support_b = one_epoch(dataloader=test_loader, split='test', model=model, device=device, optimizer=None)

        ttl_cnt += [y_true.shape[0]]

        acc_all += [acc]
        recall_all += [recall[1]]
        precision_all += [precision[1]]
        f1_all += [f1[1]]

        acc_baseline += [acc_b]
        recall_all_baseline += [recall_b[1]]
        precision_all_baseline += [precision_b[1]]
        f1_all_baseline += [f1_b[1]]

        if verbose in {2}:
            print(i, '\t:\t', acc, f'(mean={round(np.mean(acc_all), 2)})\t', recall[1], '|\t', acc_b, len(test_loader.dataset))
        
        
        

    recall_all = np.array(recall_all)
    recall_all_baseline = np.array(recall_all_baseline)
    precision_all = np.array(precision_all)
    precision_all_baseline = np.array(precision_all_baseline)
    f1_all = np.array(f1_all)
    f1_all_baseline = np.array(f1_all_baseline)
    acc_all = np.array(acc_all)
    acc_baseline = np.array(acc_baseline)
    ttl_cnt = np.array(ttl_cnt)

    metric_names = ['precision', 'recall', 'f1', 'accuracy']
    
    ### print scores
    print('\nRAG-IM:')
    for idx, metric_all in enumerate([precision_all, recall_all, f1_all, acc_all]):
        print('\t', metric_names[idx])
        print('\t\tWeighted mean:\t', (metric_all * ttl_cnt).sum() / ttl_cnt.sum())
        print('\t\t Mean:\t\t', metric_all.mean())

    print('\nBaseline:')
    for idx, metric_all in enumerate([precision_all_baseline, recall_all_baseline, f1_all_baseline, acc_baseline]):
        print('\t', metric_names[idx])
        print('\t\tWeighted mean:\t', (metric_all * ttl_cnt).sum() / ttl_cnt.sum())
        print('\t\tMean:\t\t', metric_all.mean())

    # print('\t\t Total samples:', ttl_cnt.sum())

    return acc_all, acc_baseline, ttl_cnt


# In[140]:


model = Net(c_in=768, m_in=218, embed_dim=512, attn_dropout=0.0, num_attn_heads=4, num_blocks=3).to(device)
model.load_state_dict(torch.load('model_rag_im.pth', map_location=torch.device(device)))
model.eval()

ranges_all = [[0, 342], [342, 1522], [1522, 2307], [2307, 4412], [4412, len(X_tests_1)]]

print('Starting evaluation on different task groups...')
for idx, (start, end) in enumerate(ranges_all):
    print('\n')
    print(f'Task IDs in: [{start}, {end})')
    if idx == 3:
        print('\t[Few-Shot Regime]')
    elif idx == 4:
        print('\t[Zero-Shot Regime]')
    acc_all, acc_baseline, ttl_cnt = eval_all(start=start, end=end, verbose=0)
    pickle.dump([acc_all, acc_baseline, ttl_cnt], open(f'{result_dir}/acc_all__acc_baseline__ttl_cnt__start:{start}_end:{end}.pkl', 'wb'))

print('Done infererence!')