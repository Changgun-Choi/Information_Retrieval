# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 17:18:02 2021

@author: ChangGun Choi
"""


#%%
import pandas as pd
import numpy as np

path="C:/Users/ChangGun Choi/Desktop/0. 수업자료/0. IR project/File/Train_triples"

# to concatenate positive and negative examples ("split up" the data set)

train_pos = pd.read_csv(path + "/triples.train.small.tsv",
                    sep = "\t", nrows = 100, header = None, skiprows = 1, usecols = [0,1])  # [1]
train_neg = pd.read_csv(path + "/triples.train.small.tsv",
                    sep = "\t", nrows = 100, header = None, skiprows = 1, usecols = [0,2])
# FULL DATA
#train_pos = pd.read_csv(path + "/triples.train.small.tsv",
#                    sep = "\t", header = None, skiprows = 1, usecols = [0,1])  # [1]
#train_neg = pd.read_csv(path + "/triples.train.small.tsv",
#                    sep = "\t", header = None, skiprows = 1, usecols = [0,2])

train_pos.columns = ["query", "passage"]
train_neg.columns = ["query", "passage"]
train_pos["relevant"] = 1 # target label
train_neg["relevant"] = 0 

train = train_pos.append(train_neg)   # List adding


X_train = train[["query", "passage"]]
y_train = train["relevant"]
#%%
#1) Query, Passage separate -> np.column_stack

#X_train_tokenized = np.column_stack((query_train, passage_train))
#X_train_tokenized[0]

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
import nltk

import urllib.request
#from konlpy.tag import Mecab

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
y_train = y_train.to_numpy()

#Tokenize # remove punctuation
query_train = np.empty_like(query)
for i, w in enumerate(query):
    query_train[i] = nltk.regexp_tokenize(w, '[A-Za-z]+')
    
query_train[0]    

passage_train = np.empty_like(passage)
for i, w in enumerate(passage): 
    passage_train[i] = nltk.regexp_tokenize(w, '[A-Za-z]+')
passage_train[0]

#Stemming 
s = PorterStemmer()  
stop_words = set(stopwords.words('english'))
stop_words  # already Lowered 

vocab = {}

query_stop = np.empty_like(query) #200
for i in range(200):
    result = []                 #Initialize for each i as list
    for w in query_train[i]:
        w = w.lower()           # Lowering
        if w not in stop_words: # Stopwords
            w = s.stem(w)       # PorterStemmer() 
            result.append(w) 
            
            if w not in vocab:  # Vocab Dictionary
                vocab[w] = 0    
            vocab[w] += 1
            
    query_stop[i] = result      # Give index for each i 
query_stop[:3]

 
passage_stop = np.empty_like(passage)  #200
for i in range(200):
    result = []                
    for w in passage_train[i]:
        w = w.lower()
        if w not in stop_words:
            w = s.stem(w)
            result.append(w)        
            if w not in vocab:  # Vocab Dictionary
                vocab[w] = 0    
            vocab[w] += 1                   
    passage_stop[i] = result
passage[0]
passage_stop[0]


X = np.column_stack((query_stop, passage_stop))
X[0]

#%%
# 1) NLTK FreqDist Integer Encoding
#tokenizer = Tokenizer()
#tokenizer.fit_on_texts(X)

#%%
# 2) Manual Integer Encoding
# Manual Vocab sorted  #sorted key: depoends on x[1] means 2nd index : Frequency
vocab_sorted = sorted(vocab.items(), key = lambda x:x[1] , reverse = True)
print(vocab_sorted)

vocab_size = len(vocab) #2047
# enumerate : (list, set, tuple, dictionary, string) as input -> Return index
word_to_index = {word[0] : index for index, word in enumerate(vocab_sorted)} # 0, 1 leave for later
#word_to_index['pad'] = 1 
#word_to_index['unk'] = 0      #  This part made Numpy Object  
word_to_index

X[0][0]
# Encoding 
encoded = np.empty_like(X) #200 numpy 
for i in range(200):
    for j in range(2):
        integer_index = []           #Initialize for each i as list
        for w in X[i][j]:
            try:
                integer_index.append(word_to_index[w])
            except KeyError:
                integer_index.append(word_to_index['unk'])
        
        encoded[i][j] = integer_index   # Give index for each i 
encoded[0]

#%%
################################################################################################################
y_train
#Word2Vec
# Padding
X_train = encoded
X_train[0]
#max_l=max(i, for i, j in encoded[i][0], encoded[i][1] ]
#    len(l) for l in encoded[0])
#max_l
max_len = 20
#for i in range()
#=pad_sequences(encoded, maxlen=max_len, padding='post') # padding 0
#print(X_train)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

model = Sequential()
model.add(Embedding(vocab_size, 60, input_length=max_len))   # Embedding Dimension : 4 
model.add(Flatten()) # Dense의 입력으로 넣기위함.
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(15, activation='sigmoid'))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.summary()


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])         #validation_data=(x_val, y_val),
model.fit(A_X, A_Y, batch_size = 64, validation_split= 0.2, epochs=100, verbose=2) #validation_split= 0.2                     

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)
weights

#%%
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
#tokenizer = Tokenizer()   # Keras Tokenizer                 # Easy way: fit_on_texts
#[tokenizer.fit_on_texts(query_stop[i]) for i in range(200)] # Make Vocabulary depends on Frequency
#encoded = [tokenizer.texts_to_sequences(query_stop[i]) for i in range(200)]  # Indexing 
#encoded[0]



##%% EValuate

#%% # Word2Vec


from gensim.models import Word2Vec, KeyedVectors
model = Word2Vec(sentences=result, size=100, window=5, min_count=5, workers=4, sg=0)


################ HERE UP TO NOW!!! ######################################

#%%   ###################################################
Evaluation 
# top1000.dev.tsv 
# top1000.eval.tsv 
# qrels.dev.small.tsv














