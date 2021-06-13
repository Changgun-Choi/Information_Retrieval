# -*- coding: utf-8 -*-
"""
Created on Sun May 23 17:49:33 2021

@author: Moritz
"""

import pickle
import statsmodels.api as sm # better summary table than sklearn's version
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import random


def save_obj(obj, name ):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)
        
def load_eval_data(name):
    with open(path_data + name + '.pkl', 'rb') as f:
        return pickle.load(f)   

#path_data = "E:/University/Information_Retrieval_Project/data/pairwise/"
#path = "D:/MXLinux/Information_Retrieval_Project/data/"

path_data = "/media/Moritz/Seagate Backup Plus Drive/University/Information_Retrieval_Project/data/pairwise/"
path = "/media/Moritz/080FFDFF509A959E/MXLinux/Information_Retrieval_Project/data/"

df = load_eval_data("train_pairwise_chunk_0")
for i in range(1,20):
    df = df.append(load_eval_data("train_pairwise_chunk_{}".format(i * 1000)), ignore_index = True)
## the first row needs to be deleted..
df = df.iloc[1:]


X = df[["bm25","bert_sim_word","bert_sim_sentence"]]
scaler = preprocessing.StandardScaler().fit(X)
save_obj(scaler, "scaler_pairwise")
X = scaler.transform(X)
X = sm.add_constant(X)
y = df["y"]

print("Covariance matrix of the variables:")
print("")
print(np.cov(X, rowvar = False)) # only mild correlations, like for the pointwise case.
print("")

logreg = sm.Logit(y,X).fit()
print(logreg.summary())
save_obj(logreg, "logreg_pairwise")

# metrics
def cross_validation_logreg(X, y, k = 5):
    indexes = []
    index_available = set(range(len(X)))
    for i in range(k):
        indexes.append(set(random.sample(index_available, int(len(X)/5))))
        index_available = index_available - indexes[i]
    auc = []
    for index in indexes:
        model = sm.Logit(y.values[list(set(range(len(X))) - index)],X[list(set(range(len(X))) - index)]).fit()
        auc.append(roc_auc_score(y.values[list(index)], model.predict(X[list(index)])))
    return auc
auc = cross_validation_logreg(X, y) 
print("AUC using cross-validation: {}".format(auc)) 


preds = logreg.predict(X)

#print("Accuracy on training set: ")
#print(accuracy_score(y, preds))
print("")
print("AUC on entire training set:")
print(roc_auc_score(y, preds))

print("")
print(logreg.summary().as_latex())
print("")

# like for pointwise, test for interaction effects:
    
    
X_inter = df[["bm25","bert_sim_word","bert_sim_sentence"]].copy()
X_inter["bm25_bert_sim"] = X_inter["bm25"] * X_inter["bert_sim_word"]
X_inter["bm25_bert_sentence_sim"] = X_inter["bm25"] * X_inter["bert_sim_sentence"]
X_inter["bert_sim_bert_sentence_sim"] = X_inter["bert_sim_word"] * X_inter["bert_sim_sentence"]
X_inter["all"] = X_inter["bert_sim_word"] * X_inter["bm25"] * X_inter["bert_sim_sentence"]
scaler2 = preprocessing.StandardScaler().fit(X_inter)
X_inter = scaler2.transform(X_inter) # Interaction attributes are scaled after multiplication.

logreg_inter = sm.Logit(y, X_inter).fit()
print(logreg_inter.summary())

## Observation: interaction effects do not seem to play a role here, 
## maybe except for the interaction of all variables.



