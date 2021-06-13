#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from multiprocessing import Pool
import pandas as pd
import numpy as np
import pickle
import sys
sys.path.append('/media/Moritz/080FFDFF509A959E/BWsync_share/Master_BW/Information_Retrieval_Project/code')
from preprocess import preprocess


path = "/media/Moritz/Seagate Backup Plus Drive/University/Information_Retrieval_Project/data/"


### This file loads the passage corpus, preprocesses the passages using the file "preprocess.py", 
### and creates a dictionnary with term - idf(term) as term-value pairs. 
### Different chunks are read in and saved to the hard drive. This process is performed on 5 CPUs in parallel.
### These chunks are then combined into a single idf file. 

### Note that this code cannot be run directly as it is (several indexes need to be adjusted). 



def save_obj(obj, name ):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)
        
def load_obj(name):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)

start_line = 8840000
chunksize = 50
chunksize_process = 10000
start_lines = np.arange(start_line, chunksize_process * 5 + start_line, chunksize_process) 
savesize = 10000
end_lines = start_lines + chunksize_process
print_index = [False, False, False, False, True]


def inverted_index_creation(start_end):
    start_line, end_line, print_index = start_end
    inverted_index = {}

    for i in np.arange(start_line, end_line, chunksize):
        
        start_time = time.time()
        # read chunk
        passages = pd.read_csv(path + "collection.tsv",
                        sep = "\t", nrows = chunksize, skiprows = i, header = None)
    
        # tokenize
        passages_bow = np.array([passages[0], passages[1].apply(preprocess)]).transpose()
        
        
        ### create dictionnary with postings.
        unique_words= set([item for sublist in passages_bow.transpose()[1] for item in sublist])
    
        for word in list(unique_words):
            occurs_in_passages = []
            for l in range(len(passages_bow)):
                if word in passages_bow[l][1]: 
                    occurs_in_passages.append(passages_bow[l][0]) # append passage id
            try:
                inverted_index[word]["passages"].extend(occurs_in_passages)
            except KeyError:
                #inverted_index.update({word:{"tf": count, "idf": 0, "passages": occurs_in_passages}})
                # "idf" is 0, because the value can only be calculated based on all chunks
                inverted_index.update({word:{"passages": occurs_in_passages}})
                
        # delete variables to free memory
        del(passages)
        del(passages_bow)
            
        if (i % savesize == 0) and (i != start_line):
            save_obj(inverted_index, "/inverted_index/parallel/chunk{}".format(i))
            del(inverted_index)
            inverted_index = {}
        if print_index:
            print(i, time.time() - start_time)
    # for the last chunk:
    save_obj(inverted_index, "/inverted_index/parallel/chunk{}".format(end_line))  
    del(inverted_index)
    
def inverted_index_end():
    
    inverted_index = {}
    
    passages = pd.read_csv(path + "collection.tsv",
                    sep = "\t", skiprows = 8840000, header = None)
    # tokenize
    passages_bow = np.array([passages[0], passages[1].apply(preprocess)]).transpose()
    
    ### create dictionnary with postings.
    unique_words= set([item for sublist in passages_bow.transpose()[1] for item in sublist])
    for word in list(unique_words):
        occurs_in_passages = []
        for l in range(len(passages_bow)):
            if word in passages_bow[l][1]: # TODO: not only occurence, but how many times?
                occurs_in_passages.append(passages_bow[l][0]) # append passage id
        try:
            inverted_index[word]["passages"].extend(occurs_in_passages)
        except KeyError:
            #inverted_index.update({word:{"tf": count, "idf": 0, "passages": occurs_in_passages}})
            # "idf" is 0, because the value can only be calculated based on all chunks
            inverted_index.update({word:{"passages": occurs_in_passages}})
    save_obj(inverted_index, "/inverted_index/parallel/chunk{}".format(8841823))  
    del(inverted_index)

inverted_index_end()

def main():


    print(f'starting computations on 5 cores')

    with Pool() as pool:
        pool.map(inverted_index_creation, list(zip(start_lines,end_lines, print_index)) )
    
    



if __name__ == '__main__':
    main()


#%%
### all the different chunks saved (most based on 50000 passages) will now be merged into a single dictionnary.


chunk_indexes = np.append(np.arange(50000,8800000, 50000), np.array([8800000, 8810000, 8820000, 8830000, 8840000, 8841823]))
loop_indexes = [0,50,100,150,181]
for i in range(4):
    inverted_index = {}
    for index in chunk_indexes[loop_indexes[i]:loop_indexes[i+1]]:
        chunk = load_obj("/inverted_index/parallel/chunk{}".format(index))
        for item in list(chunk.items()):
            try: 
                inverted_index[item[0]]["passages"].extend(item[1]["passages"])
            except KeyError:
                inverted_index.update({item[0]:item[1]})
        save_obj(inverted_index,"/inverted_index/inverted_index_chunk_{}".format(i))
        print(index)
    del(chunk)
    del(inverted_index)
    
#%%


tf = {}
for i in range(4):
    chunk = load_obj("/inverted_index/inverted_index_chunk_{}".format(i))
    for item in list(chunk.items()):
        try: 
            tf[item[0]] += len(item[1]["passages"])
        except KeyError:
            tf.update({item[0]:len(item[1]["passages"])})
    print(i)
    del(chunk)

for term in tf:
    tf[term] = np.log10(8841823/tf[term])



#%%
some_keys = list(tf.keys())
x = np.zeros((1000000))
for i in range(1000000):
    x[i] = tf[some_keys[i]]

keys = tf.keys()
values = tf.values()

#save_obj(keys, "keys")
#save_obj(values, "values")

idf = dict(zip(keys,values))
del(tf)


#%%

idf = load_obj("idf")
for term in idf.keys():
    idf[term] = np.round(np.log10(8841823/idf[term]), decimals = 4)
#idf["attempt"]    
    
save_obj(idf,"idf")
    
