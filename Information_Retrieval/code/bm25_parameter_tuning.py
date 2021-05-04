import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
import sys
import pickle 

sys.path.append('/media/Moritz/080FFDFF509A959E/BWsync_share/Master_BW/Information_Retrieval_Project/code')
from preprocess import preprocess # own preprocessing function

path = "/media/Moritz/Seagate Backup Plus Drive/University/Information_Retrieval_Project/data/"


# to concatenate positive and negative examples ("split up" the data set)
# read positive and then negative entries
train_pos = pd.read_csv(path + "triples.train.small.tsv",
                    sep = "\t", nrows = 200000, header = None, skiprows = 1, usecols = [0,1])
train_neg = pd.read_csv(path + "triples.train.small.tsv",
                    sep = "\t", nrows = 200000, header = None, skiprows = 1, usecols = [0,2])  


def load_obj(name):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)
idf = load_obj("idf")

#%% Data manipulation


train_pos.columns = ["query", "passage"]
train_neg.columns = ["query", "passage"]
train_pos["relevant"] = 1 # target label
train_neg["relevant"] = 0

train = train_pos.append(train_neg)



    
#%% Feature computation

### define functions to extract features given a query and passage

def bm25(idf, query, passage, avg_len_passages, k = 1.5, b = 0.75): # what is a common value for k?
    
    query_bow = preprocess(query)
    passage_bow = preprocess(passage)
    common_words = list(set(query_bow) & set(passage_bow))
    bm25 = 0
    for word in common_words: 
        bm25 += (idf[word] * (k + 1) * np.count_nonzero(passage_bow == word) 
                 / (np.count_nonzero(passage_bow == word) + k * ((1 - b) + b * passage_bow.size/avg_len_passages)))
    return bm25

### add bm25 of query, passage pair as a feature to the dataset:
X = train.copy()
avg_len_passages = 35.4 # See file "estimate_avg_passage_len.py"
#X["bm25"] = X[["query", "passage"]].apply(lambda x: bm25(idf, x[0], x[1], avg_len_passages),axis=1)


#%%
# define y
range_b = np.arange(0.5,1,0.1)
range_k = np.arange(1,2,0.1)
#performance_grid = pd.DataFrame(np.zeros((len(range_b),len(range_k))))
#performance_grid = performance_grid.reindex(np.round(list(range_b), 3))
#performance_grid.columns = np.round(list(range_k),3)
y = X["relevant"]

# approach: Since we know that b = 0.75 is a common choice, 
# we choose an optimal k keeping b fixed at 0.75 and then optimize b for this k.
auc_b_075 = []
b = 0.75
for k in range_k:
	#compute X based on b and k	
    x = X[["query", "passage"]].apply(lambda x: bm25(idf, x[0], x[1], avg_len_passages, k = k, b = b), axis=1)
    model = LogisticRegressionCV(cv = 5).fit(x.values.reshape(-1,1), y)
    pred = model.predict_proba(x.values.reshape(-1,1)).transpose()[1]
    auc = metrics.roc_auc_score(y,pred)
    auc_b_075.append(auc)
    print("for k = {} and b = {} auc is {}".format(k,b, auc))
          
k_opt = list(range_k)[np.argmax(auc_b_075)]

auc_k_opt = []
for b in range_b: 
    x = X[["query", "passage"]].apply(lambda x: bm25(idf, x[0], x[1], avg_len_passages, k = k_opt, b = b), axis=1)
    model = LogisticRegressionCV(cv = 5).fit(x.values.reshape(-1,1), y)
    pred = model.predict_proba(x.values.reshape(-1,1)).transpose()[1]
    auc = metrics.roc_auc_score(y,pred)
    auc_k_opt.append(auc)
    print("for k = {} and b = {} auc is {}".format(k_opt, b, auc))

b_opt = list(range_b)[np.argmax(auc_k_opt)]
print("Highest auc of {} achieved with b = {} and k = {}".format(max(auc_k_opt), b_opt, k_opt))

# compare to k_opt with b = 0.75 
x = X[["query", "passage"]].apply(lambda x: bm25(idf, x[0], x[1], avg_len_passages, k = k_opt, b = 0.75), axis=1)
model = LogisticRegressionCV(cv = 5).fit(x.values.reshape(-1,1), y)
pred = model.predict_proba(x.values.reshape(-1,1)).transpose()[1]
auc = metrics.roc_auc_score(y,pred)
print("for k = {} and b = {} auc is {}".format(k_opt, 0.75, auc))

# Observation: auc is not stringly effected by the choices of k and b. Highest auc is achieved for 
# k = 1, b = 0.8, so these hyperparameters will be used in the feature computation.
