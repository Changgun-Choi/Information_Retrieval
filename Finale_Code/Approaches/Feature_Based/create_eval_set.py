# -*- coding: utf-8 -*-
"""
Created on Wed May 26 18:52:30 2021

@author: Moritz
"""

import time
from multiprocessing import Pool
import pandas as pd
import numpy as np
import pickle
import sys
sys.path.append('D:/MXLinux/Information_Retrieval_Project/code')
from preprocess import preprocess # own preprocessing function, own file
from functions import bm25, cos_sim_sentence_embeddings, similarity_bert_word_embeddings

path = "E:/University/Information_Retrieval_Project/data/"


def load_obj(name):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)
def save_obj(obj, name ):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)
        
idf = load_obj("idf")

for chunk_index in range(200,201):
    
    start = time.time()
    df = pd.read_csv(path + "all_dev3.csv", sep = "\t", nrows = 1000, skiprows = chunk_index * 1000, header = None)
    df.columns = "qid pid query passage".split(" ")
    # calculate bm25
    df["bm25"] = df[["query","passage"]].apply(lambda x: bm25(idf, x.iloc[0], x.iloc[1], 35.4),axis=1)
    # calculate bert_word_embeddings similarity
    df["bert_sim_word"] = df[["query", "passage"]].apply(lambda x: similarity_bert_word_embeddings(x.iloc[0], x.iloc[1], idf), axis = 1)
    # calculate bert sentence embeddings similarity
    df["bert_sim_sentence"] = df[["query", "passage"]].apply(lambda x: cos_sim_sentence_embeddings(x.iloc[0], x.iloc[1]), axis = 1) 
    save_obj(df[["qid","pid","bm25","bert_sim_word", "bert_sim_sentence"]], "eval_data_features/all_features_{}".format(chunk_index * 1000))
    print("iteration {}: {}".format(chunk_index * 1000, time.time() - start))
    
    
##x = load_obj("eval_data_features/all_features_0")

    


    