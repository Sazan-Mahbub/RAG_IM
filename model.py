from utils import *

class Net(nn.Module):
    def __init__(self, c_in, m_in, embed_dim, attn_dropout, num_attn_heads, num_blocks):
        super(Net, self).__init__()
        # self.logRegDict = torch.nn.Parameter(logRegModels)  ## Dictionary of all logestic regression models.
        self.c_in = c_in
        self.m_in = m_in
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.num_attn_heads = num_attn_heads
        self.num_blocks = num_blocks
        
        self.linear_in_query   = nn.Linear(c_in, embed_dim)
        self.linear_in_key = nn.Linear(c_in+m_in, embed_dim)
        self.linear_in_value = nn.Linear(m_in, embed_dim)
        
        self.multihead_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim,
                                  num_heads=num_attn_heads, 
                                  batch_first=True, 
                                  dropout=attn_dropout) for _ in range(num_blocks)
        ])
        self.bottleneck = nn.ModuleList([
                nn.Sequential(
                nn.Linear(embed_dim, embed_dim//2),
                nn.Dropout(attn_dropout),
                nn.ELU(inplace=True),
                nn.Linear(embed_dim//2, embed_dim),
                nn.Dropout(attn_dropout),
                nn.ELU(inplace=True),
                # nn.Linear(embed_dim, m_in),
                # nn.Dropout(attn_dropout),
                # nn.ELU(inplace=True),
            ) for _ in range(num_blocks)
        ])
        # self.update_gate = nn.Sequential(
        #     # nn.Linear(m_in*2, m_in),
        #     # nn.Dropout(attn_dropout),
        #     # nn.ELU(inplace=True),
        #     nn.Linear(m_in*2, 1),
        #     nn.Sigmoid()
        # )
        self.update_gate = torch.nn.Parameter(torch.tensor(0.5))
        self.linear_out = nn.Linear(embed_dim, m_in)

    def forward(self, c, x, y, self_model, nbr_models, nbr_embs):
        self_model = self_model.unsqueeze(1)
        query = c.unsqueeze(1) #torch.cat([self_model, c], -1)  ## assumption: for a new task, we may not have a linear model, or we may have a very bad one
        key = torch.cat([nbr_models, nbr_embs], -1)  ## assumption: we will choose neighbors that have high confidense (probably saw more training samples!)
        value = nbr_models  ## assumption: we want to mix the logreg model weights. MHA mixes the value vectors. So we set value as the neighbor logreg models.
        query = self.linear_in_query(query) # NOTE: query (N x 1 x E)
        key = self.linear_in_key(key)       # NOTE: key (N x S x E)
        value = self.linear_in_value(value) # NOTE: val (N x S x E)
        # self_model = self.linear_in_value(self_model) # NOTE: val (N x 1 x E)
        # print(self_model.shape);exit()
        
        # print(query.shape, key.shape, value.shape)
        
        for b_idx, attn_layer in enumerate(self.multihead_attn):
            attn_output, attn_output_weights = attn_layer(query=query, key=key, value=value, 
                                                          need_weights=True, average_attn_weights=True,)

            ##@ bottleneck ffn & residual connection
            # attn_output = (attn_output + self_model) / 2
            query = (query + self.bottleneck[b_idx](attn_output)) / 2   # NOTE: (N x 1 x E)

        attn_output = self.linear_out(attn_output)   # NOTE: (N x 1 x m_in)

        ## TODO: debug only
        # attn_output = F.tanh(attn_output) * 0.5 + F.tanh(self_model) * 0.5
        update_probab = self.update_gate #self.update_gate(torch.cat([attn_output, self_model], -1))
        attn_output = attn_output * (update_probab) + self_model * (1 - update_probab)

        # y_pred_logits = torch.bmm(attn_output, x.unsqueeze(-1)).view(-1)   # NOTE: (N x 1 x m_in) X (N x m_in x 1) => (N x 1 x 1) => (N)
        y_pred_logits = torch.diag(attn_output.squeeze(1) @ x.T).view(-1)
        y_pred = F.sigmoid(y_pred_logits)
        loss = F.binary_cross_entropy(y_pred, y)

        return y_pred, loss




class DecoupledLogisticRegression:

    def __init__(self, classif_model: Callable, n_classes: int) -> None:
        """
        Args:
            classif_model (None -> model type): wrapper function that returns
                instance of classification model with `fit`, `model_decision_function`
                class methods for training and logit outputs
            n_classes (int): number of potential class labels, >0
        """
        self.n_classes = n_classes
        self.classif_model = classif_model
        self.models_binary_decoupled = []

        self.check_rep()

    def fit(self, X_train_k, Y_train_k):
        """
        Trains `self.n_classes` binary log. regression models
        where the kth model predicts OvR (one vs. rest) probability for class k

        Args:
            X_train_k list(nd.array): list of (n_train_k, d_features) training features for each of `n_classes`
            Y_train_k list(nd.array): list of (n_train_k, ) training features for each of `n_classes`
        Returns:
            None
        """

        # Train each decoupled model

        model_k = self.classif_model()
        model_k.fit(X_train_k, Y_train_k)

        self.models_binary_decoupled.append(model_k)

        self.check_rep()

    def predict(self, X, k):
        """
        Args:
            X (nd.array, pd.df, pd.Series): features of size (n_samples, d)
            k (int): 0 <= k < n_classes denoting the class to predict P_k(Y_i = 1 | X_i)
        Returns:
            nd.array: (n_samples,) binary predictions for each sample
        """
        assert 0 <= k < self.n_classes

        model_k = self.models_binary_decoupled[k]
        Y_pred = model_k.predict(X) # (n_samples, ) binary logits for each sample

        self.check_rep()

        return Y_pred

    def predict_proba(self, X, k):
        """
        Args:
            X (nd.array, pd.df, pd.Series): features of size (n_samples, d)
            k (int): 0 <= k < n_classes denoting the class to predict P_k(Y_i = 1 | X_i)
        Returns:
            nd.array: (n_samples,) binary predictions for each sample
        """
        model_k = self.models_binary_decoupled[k]
        Y_proba = model_k.decision_function(X) # (n_samples, ) probabilities for each sample

        self.check_rep()
        return Y_proba

    def check_rep(self):
        assert self.n_classes > 0

def create_log_reg_model():
    return LogisticRegression(max_iter=3000, multi_class='multinomial', solver='lbfgs')



