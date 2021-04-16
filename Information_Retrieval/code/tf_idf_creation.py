#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 21:49:12 2021

@author: Moritz
"""

### This file creates a dictionnary containing the number of passages (documents) in which a term occurs 
### (not the total number of appearances!) and the idf of this term.
### The file is not ready to run, it is so far only run sequentially with different parameters...


import pandas as pd
import numpy as np

import time
import pickle
from preprocess import preprocess

path = "/media/Moritz/Seagate Backup Plus Drive/University/Information_Retrieval_Project/data/"


def save_obj(obj, name ):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)


### read data sequentially. Lines from start_index to end_line are read.

inverted_index = {}

chunksize = 50
end_line = 3000000 
savesize = 50000 
start_index = 2000000

# past saves: 0 - 2 000 000
#%%

start = time.time()

for i in np.arange(start_index, end_line, chunksize):
    
    step_1 = time.time()
    
    # read chunk
    passages = pd.read_csv(path + "collection.tsv",
                    sep = "\t", nrows = chunksize, skiprows = i, header = None)

    # tokenize
    passages_bow = np.array([passages[0], passages[1].apply(preprocess)]).transpose()
    
    
    ### create dictionnary with term frequencies. Code for postings greyed out.
    unique_words= set([item for sublist in passages_bow.transpose()[1] for item in sublist])

    for word in list(unique_words):
        count = 0
        #occurs_in_passages = []
        for l in range(len(passages_bow)):
            if word in passages_bow[l][1]: # TODO: not only occurence, but how many times?
                #occurs_in_passages.append(passages_bow[l][0]) # append passage id
                count += 1
        try:
            inverted_index[word]["tf"] += count
            #inverted_index[word]["passages"].extend(occurs_in_passages)
        except KeyError:
            #inverted_index.update({word:{"tf": count, "idf": 0, "passages": occurs_in_passages}})
            # "idf" is 0, because the value can only be calculated based on all chunks
            inverted_index.update({word:{"tf": count, "idf": 0}})
            
    # delete variables to free memory
    del(passages)
    del(passages_bow)
    
    step_2 = time.time()
    print("{}: {} seconds".format(i, step_2 - step_1))
    
    if (i % savesize == 0) and (i != start_index):
        save_obj(inverted_index, "/inverted_index/tf_idf{}".format(i))
        del(inverted_index)
        inverted_index = {}
        
# for the last chunk:
save_obj(inverted_index, "/inverted_index/tf_idf{}".format(end_line))  
del(inverted_index)
end = time.time()
print(end-start)


#%%

term_frequencies = {}

for i in np.append(np.arange(savesize, end_line, savesize), end_line):
    index_keys = term_frequencies.keys()
    chunk = load_obj("/inverted_index/tf_idf{}".format(i))
    for key in chunk.keys():
        try:
            term_frequencies[key]["tf"] += chunk[key]["tf"]
        except KeyError:
            term_frequencies.update({key:{"tf":chunk[key]["tf"], "idf": 0}})
    print("chunk {} is read".format(i))
    del(index_keys)
    del(chunk)

# idf: take 8,841,823 total passages, but we only used "end_line" in this iteration.. so use "end_line"
# this enables us to use at least an approximate of the idf values.
 
for key in term_frequencies.keys():
    term_frequencies[key]["idf"] = np.log((end_line-start_index)/term_frequencies[key]["tf"])


save_obj(term_frequencies, "/inverted_index/tf_idf_approximated_{}".format(end_line))



