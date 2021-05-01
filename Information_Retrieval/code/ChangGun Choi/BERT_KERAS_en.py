# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 11:09:05 2021

@author: ChangGun Choi
"""
import tensorflow as tf
print(tf.__version__)
import pandas as pd
import numpy as np
import re
import pickle
from tensorflow import keras
import keras as keras
from keras.models import load_model
from keras import Input, Model
from keras import optimizers
from keras.layers import Layer
import codecs
from tqdm import tqdm
import shutil
#%%
from keras_bert import load_trained_model_from_checkpoint, load_vocabulary
from keras_bert import Tokenizer
from keras_bert import AdamWarmup, calc_train_steps

from keras_radam import RAdam

#%%
import pandas as pd
import numpy as np

path="C:/Users/ChangGun Choi/Desktop/0. 수업자료/0. IR project/File/Train_triples"

# to concatenate positive and negative examples ("split up" the data set)
                      #sep : separating
train_pos = pd.read_csv(path + "/triples.train.small.tsv",
                    sep = "\t", nrows = 10, header = None, skiprows = 1, usecols = [0,1])  # [1]
train_neg = pd.read_csv(path + "/triples.train.small.tsv",
                    sep = "\t", nrows = 10, header = None, skiprows = 1, usecols = [0,2])
# FULL DATA
#train_pos = pd.read_csv(path + "/triples.train.small.tsv",
#                    sep = "\t", header = None, skiprows = 1, usecols = [0,1])  # [1]
#train_neg = pd.read_csv(path + "/triples.train.small.tsv",
#                    sep = "\t", header = None, skiprows = 1, usecols = [0,2])
type(train_pos)
train_pos.columns = ["query", "passage"]
train_neg.columns = ["query", "passage"]
train_pos["relevant"] = 1 # target label
train_neg["relevant"] = 0 
train_pos
train_neg

#train = train_pos.append(train_neg)   # List adding
train = pd.concat([train_pos, train_neg])
train.reset_index(inplace=True, drop=True)   # Resetting Index
train
y_train = train["relevant"]
#%%

SEQ_LEN = 400 

BATCH_SIZE = 10         # Small for CPU, GPU        
EPOCHS=2 
LR=1.5e-5                                                                            

pretrained_path = "C:/Users/ChangGun Choi/Desktop/0. 수업자료/0. IR project/BERT/uncased_L-12_H-768_A-12"
config_path = pretrained_path + "/bert_config.json"

# {
#  "attention_probs_dropout_prob": 0.1, 
#  "directionality": "bidi", 
#  "hidden_act": "gelu", 
#  "hidden_dropout_prob": 0.1, 
#  "hidden_size": 768, 
#  "initializer_range": 0.02, 
#  "intermediate_size": 3072, 
#  "max_position_embeddings": 512, 
#  "num_attention_heads": 12, 
#  "num_hidden_layers": 12, 
#  "pooler_fc_size": 768, 
#  "pooler_num_attention_heads": 12, 
#  "pooler_num_fc_layers": 3, 
#  "pooler_size_per_head": 128, 
#  "pooler_type": "first_token_transform", 
#  "type_vocab_size": 2, 
#  "vocab_size": 119547
# }


checkpoint_path = pretrained_path + "/bert_model.ckpt"
vocab_path = pretrained_path + "/vocab.txt"

DATA_COLUMN = "passage"
QUESTION_COLUMN = "query"
RELEVANT = "relevant"
#%%

from transformers import BertTokenizer, TFBertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#encoded_input = tokenizer(text, return_tensors='tf')
#output = model(encoded_input)

#print(tokenizer.tokenize("my dog is cute", "he likes playing"))
print(tokenizer.encode("my dog is cute", "he likes playing"))
#%% #BERT Tokenizer
#token_dict = {}
#with codecs.open(vocab_path, 'r', 'utf8') as reader:
#    for line in reader:
#        token = line.strip()
#        token_dict[token] = len(token_dict)
#tokenizer = Tokenizer(token_dict)  
#reverse_token_dict = {v : k for k, v in token_dict.items()}
#%%
print("relevant : ", train.loc[0,'relevant'])

print(tokenizer.encode(train.loc[0,'query'],train.loc[0,'passage'] ))

#%%
def convert_data(data_df):
    global tokenizer
    indices, segments, masks, targets = [], [], [] ,[]
    
    for i in tqdm(range(len(data_df))):
   
        targets.append(data_df["relevant"][i])

        que = tokenizer.encode(data_df["query"][i])
        doc = tokenizer.encode(data_df["passage"][i])
        
        doc.pop(0) 
        
    
        if len(que+doc) > SEQ_LEN:  #400 
          while len(que+doc) != SEQ_LEN:
            doc.pop(-1)  #
          doc.pop(-1)
          
          
          doc.append(102) #[SEP]
        # question, context, padding
        # 00000000, 1111111, 0000000
        
        segment = [0]*len(que) + [1]*len(doc) + [0]*(SEQ_LEN-len(que)-len(doc))
        if len(que + doc) <= SEQ_LEN:
          mask = [1]*len(que+doc) + [0]*(SEQ_LEN-len(que+doc))
        else:
          mask = [1]*len(que+doc)
          
        if len(que + doc) <= SEQ_LEN:
          while len(que+doc) != SEQ_LEN: #  padding
            doc.append(0)  #passage  padding
                              
        ids = que + doc

        indices.append(ids)
        segments.append(segment)
        masks.append(mask)
        
    #  numpy array 
    indices = np.array(indices)
    segments = np.array(segments)
    masks = np.array(masks)
    targets = np.array(targets)

    return [indices, masks, segments], targets



def load_data(pandas_dataframe):
    data_df = pandas_dataframe
    data_df["query"] = data_df["query"].astype(str)     # String
    data_df["passage"] = data_df["passage"].astype(str)
    data_x, data_y = convert_data(data_df)

    return data_x, data_y   # [indices,masks,segments], targets

#%%
train_x, train_y = load_data(train)


#%%
import tensorflow_addons as tfa
opt = tfa.optimizers.RectifiedAdam(lr=1.0e-5, weight_decay=0.0025, warmup_proportion=0.05)


def get_bert_finetuning_model():
    
  model = TFBertModel.from_pretrained("bert-base-uncased")

  token_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_word_ids')
  mask_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_masks')
  segment_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_segment')

  bert_outputs = model([token_inputs, mask_inputs, segment_inputs])

  bert_outputs = bert_outputs[1]
  sentiment_first = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(bert_outputs)
  BertRanking_model = tf.keras.Model([token_inputs, mask_inputs, segment_inputs], sentiment_first)

  BertRanking_model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])
  
  return BertRanking_model

#%%
BertRanking_model = get_bert_finetuning_model()
Bert_initial_weight = BertRanking_model.weights
Bert_initial_weight

BertRanking_model.summary()
BertRanking_model.fit(train_x, train_y, epochs=4, shuffle=True, batch_size=100, validation_split= 0.2)

#%%

BertRanking_model.fit(train_x, train_y, epochs=4, shuffle=True, batch_size=100 )


#%%

weights = BertRanking_model.weights
BertRanking_model.save_weights(pretrained_path+"/bert.pointwise")

bert_model = get_bert_finetuning_model(model)
path = "gdrive/My Drive/Colab Notebooks/squad"
bert_model.load_weights(path+"/(Uncased)Squad.h5"

                        




# Next step

#%% F-1 Score



def predict_convert_data(data_df):
    global tokenizer
    tokens, masks, segments = [], [], []
    
    for i in tqdm(range(len(data_df))):

        token = tokenizer.encode(data_df[DATA_COLUMN][i], max_length=SEQ_LEN, truncation=True, padding='max_length')
        num_zeros = token.count(0)
        mask = [1]*(SEQ_LEN-num_zeros) + [0]*num_zeros
        segment = [0]*SEQ_LEN

        tokens.append(token)
        segments.append(segment)
        masks.append(mask)

    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)
    return [tokens, masks, segments]


def predict_load_data(pandas_dataframe):
    data_df = pandas_dataframe
    data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)
    data_x = predict_convert_data(data_df)
    return data_x



#%% TEST SET????

preds = sentiment_model.predict(test_set)
preds

from sklearn.metrics import classification_report
y_true = test['label']
# F1 Score 확인
print(classification_report(y_true, np.round(preds,0)))











