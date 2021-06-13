#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 16:50:48 2021

@author: Moritz
"""

import pandas as pd
import numpy as np
import pickle
import sys
sys.path.append('D:/MXLinux/Information_Retrieval_Project/code')
from preprocess import preprocess # own preprocessing function, own file
#from bert_embeddings import similarity_bert_word_embeddings #pretrained bert embeddings, own file
from functions import bm25, similarity_bert_word_embeddings, cos_sim_sentence_embeddings


### This file creates the pointwise training data.


#%% Read data

path = "E:/University/Information_Retrieval_Project/data/"


# to concatenate positive and negative examples ("split up" the data set)
# read positive and then negative entries
train_pos = pd.read_csv(path + "triples.train.small.tsv",
                    sep = "/t", nrows = 10000, header = None, skiprows = 1, usecols = [0,1])
train_neg = pd.read_csv(path + "triples.train.small.tsv",
                    sep = "/t", nrows = 10000, header = None, skiprows = 1, usecols = [0,2])  


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



### add bm25 of query, passage pair as a feature to the dataset:
X = train.copy()
avg_len_passages = 35.4 # See file "estimate_avg_passage_len.py"
X["bm25"] = X[["query", "passage"]].apply(lambda x: bm25(idf, x[0], x[1], avg_len_passages),axis=1)

### add bert embedding similarity
X["bert_sim"] = X[["query", "passage"]].apply(lambda x: similarity_bert_word_embeddings(x[0], x[1], idf), axis = 1)

### add SBERT embedding similarity
X["bert_sentence_sim"] = X[["query", "passage"]].apply(lambda x: cos_sim_sentence_embeddings(x[0], x[1]), axis = 1)


# save training data
#X.to_csv(path + "training_pointwise_200000.csv")

