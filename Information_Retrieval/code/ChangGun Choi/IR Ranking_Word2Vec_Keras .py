# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 17:18:02 2021

@author: ChangGun Choi
"""

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
train
X_train = train[["query", "passage"]]  # [[list], list]]
X_train.head()

#X_train_tokenized = np.column_stack((query_train, passage_train))
#X_train_tokenized[0]

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer

import urllib.request
#from konlpy.tag import Mecab
from nltk import FreqDist
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tensorflow_hub as hub
import warnings
warnings.simplefilter('ignore')

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#%%  # 1) Query Numpy -> Preprocessing

query = X_train["query"].to_numpy()

passage = X_train["passage"].to_numpy()
y_train = train["relevant"].to_numpy()

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
    result = []                 #Initialize for each i as list
    for w in query_train[i]:
        w = w.lower()           # Lowering
         
        if w not in stop_words: # Stopwords
            w = s.stem(w)
            result.append(w)    # Stemming 
            
            if w not in vocab:  
                vocab[w] = 0    # Vocab Dictionary
            vocab[w] += 1                      
    query_stop[i] = result      # Give index for each i 
query_stop[:3]


vocab_sorted = sorted(vocab.items(), key = lambda x:x[1] , reverse = True)
print(vocab_sorted)
vocab_sorted[0]  


# 1) NLTK FreqDist Integer Encoding
  #vocab = FreqDist(np.hstack(query_stop))
  
# 2) Manual Integer Encoding
# Manual Vocab sorted  #sorted key: 정렬기준 -> 여기서는 x의 index 1이니까 2번째 것: Frequency
vocab_sorted = sorted(vocab.items(), key = lambda x:x[1] , reverse = True)
print(vocab_sorted)
vocab_sorted[0]  
vocab_size = 263
     
# enumerate : (list, set, tuple, dictionary, string) as input -> Return index
word_to_index = {word[0] : index for index, word in enumerate(vocab_sorted)} # 0, 1 leave for later
#word_to_index['pad'] = 1 
#word_to_index['unk'] = 0      #  This part made Numpy Object  
word_to_index

# Encoding 
encoded = np.empty_like(query_stop) #200 numpy 
for i in range(200):
    integer_index = []           #Initialize for each i as list
    for w in query_stop[i]:
        try:
            integer_index.append(word_to_index[w])
        except KeyError:
            integer_index.append(word_to_index['unk'])
        
    encoded[i] = integer_index   # Give index for each i 
encoded[1]

################################################################################################################


#Word2Vec
max_len=max(len(l) for l in encoded)
X_train=pad_sequences(encoded, maxlen=max_len, padding='post')
print(X_train)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_len))   # Embedding Dimension : 4 
model.add(Flatten()) # Dense의 입력으로 넣기위함.
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(15, activation='sigmoid'))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.summary()


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])               # 더하기 validation_data=(x_val, y_val),
model.fit(X_train, y_train, batch_size = 64, validation_split= 0.2, epochs=100, verbose=2) #validation_split= 0.2                     

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)  # shape: (vocab_size, embedding_dim)


# Test 데이터를 먼저 만들어야함~~~

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(x_test[:3])
print("predictions shape:", predictions.shape)



# Integer Encoding with Keras Tokenizer      
#tokenizer = Tokenizer()   # Keras Tokenizer                 # 간단 방법: fit_on_texts
#[tokenizer.fit_on_texts(query_stop[i]) for i in range(200)] # 빈도수를 기준으로 단어 집합을 생성
#encoded = [tokenizer.texts_to_sequences(query_stop[i]) for i in range(200)]  # Indexing 
#encoded[0]











#%% EValuate




#%%





















embedding_dim=16

model = keras.Sequential([
  layers.Embedding(vocab_size, embedding_dim),
  layers.GlobalAveragePooling1D(),
  layers.Dense(32, activation='relu')
  layers.Dense(16, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.summary()



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














