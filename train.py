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
# print(logRegModels.models_binary_decoupled[0].coef_.shape, logRegModels.models_binary_decoupled[0].intercept_.shape)

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


min_sample_size = 2
selected_procs = []
i = 0
while(i < len(sorted_procedures_list) and sorted_procedures_list[i][1]['count'] >= min_sample_size):
  proc = sorted_procedures_list[i][0]
  selected_procs.append(proc)
  i += 1


# In[15]:


print(f"number of classes: {len(selected_procs)}")


# In[26]:


X_tests_1, y_tests_1, X_trains_1, y_trains_1, X_tests_2, y_tests_2, X_trains_2, y_trains_2 = pickle.load(open(f'{data_dir}/all_preprocessed_data.pkl', 'rb'))

bert_class_embeddings_df = pd.read_csv(f'{data_dir}/bert_class_embeddings.csv')
bert_class_embeddings = bert_class_embeddings_df.values[:len(X_tests_1)]
bert_class_embeddings.shape


# In[133]:
##### find top K neighbors
K = 10
cos_sim_matrix = cosine_similarity(bert_class_embeddings, bert_class_embeddings) # Calculate cosine similarity
np.fill_diagonal(cos_sim_matrix, -np.inf)
nbr_indices = np.argsort(cos_sim_matrix, axis=1)[:, -K:] # Find indices of top K closest vectors for each embedding

####### dataloaders
batch_size = 1024
train_set = CLLMdataset(split='train', X=X_trains_1, Y=y_trains_1, 
                        context_encoding=bert_class_embeddings, 
                        logreg_models=logRegModel_weights,
                        nbr_indices=nbr_indices,
                        context_idx=None)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)


model = Net(c_in=768, m_in=218, embed_dim=512, attn_dropout=0.0, num_attn_heads=4, num_blocks=3).to(device)

learning_rate = 2e-4
optim = torch.optim.AdamW(model.parameters(),
                            lr = learning_rate, # args.learning_rate - default is 5e-5.
                            eps = 1e-8, # args.adam_epsilon  - default is 1e-8.
                            weight_decay=0.01 #0.01
                         )
num_epochs = 5
for epoch in range(num_epochs):
    print("epoch:", epoch)
    one_epoch(dataloader=train_loader, split='train', model=model, device=device, optimizer=optim)
    torch.save(model.state_dict(), 'model_rag_im.pth')


