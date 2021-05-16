#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 18:15:41 2021

@author: Moritz
"""

import time
from multiprocessing import Pool
import pandas as pd
import numpy as np
import pickle
import sys
sys.path.append('/media/Moritz/080FFDFF509A959E/BWsync_share/Master_BW/Information_Retrieval_Project/code')
from preprocess import preprocess
from bm25_comp import bm25
from bert_embeddings import similarity_bert_word_embeddings

path = "/media/Moritz/Seagate Backup Plus Drive/University/Information_Retrieval_Project/data/"

def load_obj(name):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)
def save_obj(obj, name ):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)
        
idf = load_obj("idf")

qrel = pd.read_csv(path + "qrels.dev.tsv", sep = "\t", usecols=[0,2])
qrel.columns = ["qid", "pid"]
top = pd.read_csv(path + "top1000.dev", sep = "\t", usecols = [0], header = None)
top.columns = ["qid"]

qids_unique = top["qid"].unique()
len(qids_unique)

# get relevance labels
def relevance(x):
    # Input x: [qid, pid]
    # (qrel dev data has to be loaded)
    pids = qrel.loc[qrel["qid"] == x[0]]["pid"]
    if x[1] in pids.values:
        return 1
    else:
        return 0




def create_eval_set(index, chunksize, process_id):
    """
    reads in data from "top1000.dev", but only for the indexes passed.
    The data is read and saved in chunks. For every chunk, the following is performed:
        Add a column with relevance labels.
        Add columns for bm25 and bert_embeddings.
        Delete passage and query descriptions.
        Save and delete the chunk.
    
    """
    
    n_chunks = int(len(index)/chunksize)
    chunk_indexes = list(np.arange(0,n_chunks * chunksize, chunksize))
    chunk_indexes.append(chunk_indexes[len(chunk_indexes)-1] + chunksize)
    chunk_indexes.append( chunk_indexes[len(chunk_indexes)-1] + (len(index)%chunksize))
    # look f.ex. like [0, 10, 20, 30, 40, 50, 54]
    
    for i_chunk_index in range(len(chunk_indexes)-1):
        start_time = time.time()
        row_indexes = index[chunk_indexes[i_chunk_index] : chunk_indexes[i_chunk_index+1]] 
        # read part of data:
        df = pd.read_csv(path + "top1000.dev", header = None, skiprows= lambda x: x not in row_indexes, sep = "\t")
        df.columns = "qid pid query passage".split(" ")
        # add relevance labels
        df["relevant"] = df[["qid","pid"]].apply(lambda x: relevance(x), axis = 1)
        # calculate bm25
        df["bm25"] = df[["query","passage"]].apply(lambda x: bm25(idf, x.iloc[0], x.iloc[1], 35.4),axis=1)
        # calculate bert_word_embeddings
        df["bert_sim"] = df[["query", "passage"]].apply(lambda x: similarity_bert_word_embeddings(x[0], x[1], idf), axis = 1)
        
        # save data
        save_obj(df[["qid","pid","relevant", "bm25"]], "eval_data/chunk{}_p{}".format(chunk_indexes[i_chunk_index+1], process_id))
        print("{}: {}".format(chunk_indexes[i_chunk_index+1], time.time()-start_time))
        del(df)

        
n_queries = 400 # needs to be divisible by 4
rng = np.random.default_rng()
qids = rng.choice(qids_unique, size=n_queries)
index = top[top["qid"].isin(qids)].index

create_eval_set(index, 50000, 1)


"""
index_list = [index[x:x+int(len(index)/4)] for x in range(0, len(index), int(len(index)/4))]

def main():


    print('starting computations on 4 cores')

    with Pool(processes = 4) as pool:
        pool.map(create_eval_set, list(zip(index_list,[20000,20000,20000,20000], [1,2,3,4])))
    
    



if __name__ == '__main__':
    main()


"""

