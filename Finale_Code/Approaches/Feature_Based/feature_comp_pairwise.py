#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 14:41:54 2021

@author: Moritz
"""

import pandas as pd
import numpy as np
import pickle
import time
import sys
sys.path.append("/media/Moritz/080FFDFF509A959E/MXLinux/Information_Retrieval_Project/code")
from functions import bm25, similarity_bert_word_embeddings, cos_sim_sentence_embeddings
import random
from multiprocessing import Pool

#path = "/media/Moritz/Seagate Backup Plus Drive/University/Information_Retrieval_Project/data/"
path = "E:/University/Information_Retrieval_Project/data/"




def load_obj(name):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)
def save_obj(obj, name ):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

idf = load_obj("idf")   

def save_chunk(start_index, chunksize):
    start = time.time()
    chunk = pd.read_csv(path + "triples.train.small.tsv", sep = "\t", header = None, nrows = chunksize, skiprows = start_index)
    chunk.columns = ["query","passage_r","passage_nr"]
    chunk["bm25"] = chunk.apply(lambda x: bm25(idf,x.iloc[0],x.iloc[1]) - bm25(idf, x.iloc[0],x.iloc[2]), axis = 1)
    chunk["bert_sim_word"] = chunk[["query","passage_r","passage_nr"]].apply(lambda x: similarity_bert_word_embeddings(x.iloc[0], x.iloc[1], idf) - similarity_bert_word_embeddings(x.iloc[0], x.iloc[2], idf), axis = 1)
    chunk["bert_sim_sentence"] = chunk[["query","passage_r","passage_nr"]].apply(lambda x: cos_sim_sentence_embeddings(x.iloc[0], x.iloc[1]) - cos_sim_sentence_embeddings(x.iloc[0], x.iloc[2]), axis = 1)
    chunk["y"] = 1 
    ### random flipping of pairs
    random.seed(123)
    index = random.sample(range(len(chunk)), int(len(chunk)/2))
    chunk.loc[index, ["bm25","bert_sim_word","bert_sim_sentence"]] = chunk.loc[index, ["bm25","bert_sim_word","bert_sim_sentence"]] * -1
    chunk.loc[index,"y"] = chunk.loc[index,"y"] -1
    save_obj(chunk,"pairwise/train_pairwise_chunk_{}".format(start_index))
    del(chunk)
    print("row {}: {}".format(start_index, time.time()-start))

# run 2 processes
#start_index = [0,1000]
#chunksize = [1000,1000]

#save_chunk(0,1000)

for i in range(1,20):
    if i > 0:
        save_chunk(i * 1000, 1000)

#def main():
#    with Pool() as pool:
#        pool.map(save_chunk, list(zip(start_index,chunksize)) )

#if __name__ == '__main__':
#    main()



#x = load_obj("pairwise/train_pairwise_chunk_16000")
