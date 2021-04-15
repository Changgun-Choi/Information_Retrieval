# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 17:18:02 2021

@author: ChangGun Choi
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd

import tensorflow_hub as hub
import numpy as np

embed = hub.load("https://tfhub.dev/google/Wiki-words-250/2")

#%%
import pandas as pd
import numpy as np

path="C:/Users/ChangGun Choi/Desktop/0. 수업자료/0. IR project/File/Train_triples (Pair wise)"

# to concatenate positive and negative examples ("split up" the data set)
# read positive and then negative entries
train_pos = pd.read_csv(path + "/triples.train.small.tsv",
                    sep = "\t", nrows = 100, header = None, skiprows = 1, usecols = [0,1])  # [1]
train_neg = pd.read_csv(path + "/triples.train.small.tsv",
                    sep = "\t", nrows = 100, header = None, skiprows = 1, usecols = [0,2])  # [2]

train_pos.columns = ["query", "passage"]
train_neg.columns = ["query", "passage"]
train_pos["relevant"] = 1 # target label
train_neg["relevant"] = 0

train = train_pos.append(train_neg)   # List adding
#1) Query, Passage separate -> np.column_stack

X_train = train[["query", "passage"]]  # [[list], list]]
X_train.head()

#X_train_tokenized = np.column_stack((query_train, passage_train))
#X_train_tokenized[0]

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer


### PORTER STEMMER 


#%%  # 1) Query Numpy -> Preprocessing

query = X_train["query"].to_numpy()
passage = X_train["passage"].to_numpy()

query_train = np.empty_like(query)
for i, w in enumerate(query): 
    query_train[i] =  word_tokenize(w)
query_train[0]

# Vocab initialize
vocab = {}  
# from nltk import FreqDist
# vocab = FreqDist(np.hstack(sentences))
# 

s = PorterStemmer()  #Stemming 

# STOP WORDS
stop_words = set(stopwords.words('english'))
stop_words

query_stop = np.empty_like(query)  #200
for i in range(200):
    result = []                 # Initialize as list
    for w in query_train[i]:
        w = w.lower()           # Lowering
         
        if w not in stop_words: # Stopwords
            w = s.stem(w)
            result.append(w)    # Stemming 
            
            if w not in vocab:  
                vocab[w] = 0    # Vocab Dictionary
            vocab[w] += 1                      
    query_stop[i] = result      # np.array
    
query_stop[0]    
tokenizer = Tokenizer()   # Keras Tokenizer                 # 간단 방법: fit_on_texts
[tokenizer.fit_on_texts(query_stop[i]) for i in range(200)] # 빈도수를 기준으로 단어 집합을 생성
encoded = [tokenizer.texts_to_sequences(query_stop[i]) for i in range(200)]  # Indexing 

# Manual Vocab
vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True)
print(vocab_sorted)
vocab_sorted[0]

# Vocab Indexing   
 word_to_index = {word[0] : index + 1 for index, word in enumerate(vocab_sorted)}
print(word_to_index)

#%% 2) Passage  


passage_train = np.empty_like(passage)
for i, w in enumerate(passage): 
    passage_train[i] = word_tokenize(w)

passage_stop = np.empty_like(passage)  #200
for i in range(200):
    result = []                
    for w in passage_train[i]:
        w = w.lower()
        if w not in stop_words:
            result.append(w)
            if w not in vocab:
                vocab[w] = 0
            vocab[w] += 1
                
    passage_stop[i] = result
passage[0]
passage_stop[0]


#%% # Word2Vec


from gensim.models import Word2Vec, KeyedVectors
model = Word2Vec(sentences=result, size=100, window=5, min_count=5, workers=4, sg=0)




################ HERE UP TO NOW!!! ######################################












#%%  #2. List Method 
query = X_train["query"].to_list()
passage = X_train["passage"].to_list()

# Token
q_token = []
for i in range(200):
    q_token.append(word_tokenize((query[i])))

p_token = []
for i in range(200):
    p_token.append(word_tokenize((passage[i])))

# STOP WORDS
stop_words = set(stopwords.words('english'))
stop_words

result_q = []
query_stop = []
for i in range(200):
    for w in q_token[i]:
        if w not in stop_words: 
            result_q.append(w)
        query_stop.append(result_q)
        
q_token[0]
query_stop[0][100]

result_p = []
passage_stop = []
for i in range(200):
    passage_stop.append(result_p)
    for w in p_token[i]: 
        if w not in stop_words: 
            result_p.append(w)
query_stop[0][100]
passage_stop[0]




#%% 



X_train_tokenized = pd.DataFrame({"query": q_token, "passage" : p_token})
X_train_tokenized














