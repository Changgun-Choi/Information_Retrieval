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


### This file reads the training data, transforms it to a suitable format and calculates bm25 as a feature.


#%% Read data

path = "/media/Moritz/Seagate Backup Plus Drive/University/Information_Retrieval_Project/data/"


# passage data, only used to approximate the average length of a passage
passages = pd.read_csv(path + "collection.tsv",
                    sep = "\t", nrows = 1000, header = None)
number_passages = len(passages)



# to concatenate positive and negative examples ("split up" the data set)
# read positive and then negative entries
train_pos = pd.read_csv(path + "triples.train.small.tsv",
                    sep = "\t", nrows = 100, header = None, skiprows = 1, usecols = [0,1])
train_neg = pd.read_csv(path + "triples.train.small.tsv",
                    sep = "\t", nrows = 100, header = None, skiprows = 1, usecols = [0,2])  


# tf_idf for bm25: The file is not final yet! Tf, idf only based on 2 000 000 passages, not the entire corpus.
def load_obj(name):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)
tf_idf = load_obj("inverted_index/tf_idf_approximated_2000000")

#%% Data preprocessing


train_pos.columns = ["query", "passage"]
train_neg.columns = ["query", "passage"]
train_pos["relevant"] = 1 # target label
train_neg["relevant"] = 0

train = train_pos.append(train_neg)




#%%
# passages: estimate average length of passages

passages_bow = np.array([passages[0], passages[1].apply(preprocess)]).transpose()
total_number_words_in_passages = 0
for passage in passages_bow.transpose()[1]:
    total_number_words_in_passages += len(passage)
avg_len_passages = total_number_words_in_passages / number_passages

 

    
#%% Feature computation

### define functions to extract features given a query and passage

def bm25(tf_idf, query, passage, avg_len_passages, k = 1.5, b = 0.75): # what is a common value for k?
    
    query_bow = preprocess(query)
    passage_bow = preprocess(passage)
    common_words = list(set(query_bow) & set(passage_bow))
    bm25 = 0
    for word in common_words: 
        bm25 += (tf_idf[word]["idf"] * (k + 1) * np.count_nonzero(passage_bow == word) 
                 / (np.count_nonzero(passage_bow == word) + k * ((1 - b) + b * passage_bow.size/avg_len_passages)))
    return bm25

### add bm25 of query, passage pair as a feature to the dataset:
X = train.copy()
X["bm25"] = X[["query", "passage"]].apply(lambda x: bm25(tf_idf, x[0], x[1], avg_len_passages),axis=1)
