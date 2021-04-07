# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 12:07:01 2021

@author: hp
"""

import pandas as pd

#%% Read data

path = "F:/University/Information_Retrieval_Project/data/"

# to concatenate positive and negative examples ("split up" the data set)
# read positive and then negative entries
train_pos = pd.read_csv(path + "triples.train.small.tsv",
                    sep = "\t", nrows = 100, header = None, skiprows = 1, usecols = [0,1])
train_neg = pd.read_csv(path + "triples.train.small.tsv",
                    sep = "\t", nrows = 100, header = None, skiprows = 1, usecols = [0,2])  

#%% Data preprocessing

train_pos.columns = ["query", "passage"]
train_neg.columns = ["query", "passage"]
train_pos["relevant"] = 1 # target label
train_neg["relevant"] = 0

train = train_pos.append(train_neg)


#%% Feature computation

X = train.copy()

def count_chars(x):
    return len(x)

X["number_chars_q"] = X["query"].apply(count_chars)
X["number_chars_p"] = X["passage"].apply(count_chars)



