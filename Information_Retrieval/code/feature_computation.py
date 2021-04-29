# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 12:07:01 2021
@author: hp
"""

import pandas as pd
import numpy as np
import pickle
import sys
sys.path.append('/media/Moritz/080FFDFF509A959E/BWsync_share/Master_BW/Information_Retrieval_Project/code')
from preprocess import preprocess # own preprocessing function


### This file creates the training data, especially the features based on the query - passage pairs.


#%% Read data

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

def bm25(idf, query, passage, avg_len_passages, k = 1, b = 0.8): 
    # k, b determined via sequential grid search, see file "bm25_parameter_tuning.py"
    
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
X["bm25"] = X[["query", "passage"]].apply(lambda x: bm25(idf, x[0], x[1], avg_len_passages),axis=1)

X.to_csv(path + "training_pointwise_200000.csv")

#%%

#X.groupby("relevant")["bm25"].describe()

# some bad estimates:
#a = X.loc[(X["bm25"]>25) & (X["relevant"] ==0)]

#a["query"].values[2]
#a["passage"].values[2]




