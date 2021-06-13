# -*- coding: utf-8 -*-
"""
Created on Wed May 19 18:37:50 2021

@author: Moritz
"""

### This file provides functions to compute 
#       - bm25
#       - cos_similarity between embeddings based on bert word embeddings
#       - cos_similarity between embeddings based on bert sentence embeddings
# of a query, passage pair.

import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from nltk.tokenize import sent_tokenize
import torch
import sys
sys.path.append('D:/MXLinux/Information_Retrieval_Project/code')
from preprocess import preprocess # own preprocessing function, own file

path = "D:/MXLinux/Information_Retrieval_Project/data/"


def load_obj(name):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def cos_sim(x,y):
    denominator = torch.linalg.norm(x) * torch.linalg.norm(y)
    if denominator > 0:
        return torch.dot(x,y) / denominator
    else: 
        return torch.tensor(0) # nan's were discovered, so this ensures that if one sentence of the passage has embedding 0D, the entire similarity is not nan.

### bm25 
    
idf = load_obj("idf")    
    
def bm25(idf, query, passage, avg_len_passages = 35.4, k = 1, b = 0.8): 
    # k, b determined via hyperparameter tuning, see file "bm25_parameter_tuning.py"
    # average passage length estimated in another file, too.
    
    query_bow = preprocess(query)
    passage_bow = preprocess(passage)
    common_words = list(set(query_bow) & set(passage_bow))
    bm25 = 0
    for word in common_words: 
        bm25 += (idf[word] * (k + 1) * np.count_nonzero(passage_bow == word) 
                 / (np.count_nonzero(passage_bow == word) + k * ((1 - b) + b * passage_bow.size/avg_len_passages)))
    return bm25

    
### bert sentence

#Load AutoModel from huggingface model repository
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def cos_sim_sentence_embeddings(query, passage):
    #Tokenize sentences
    encoded_passage = tokenizer(passage, padding=True, truncation=True, max_length=128, return_tensors='pt')
    encoded_query = tokenizer(query, padding=True, truncation=True, max_length=128, return_tensors='pt')
    #Compute token embeddings
    
    with torch.no_grad():
        passage_emb = model(**encoded_passage)
        
    passage_emb = mean_pooling(passage_emb, encoded_passage['attention_mask'])
    with torch.no_grad():
        query_emb = model(**encoded_query)
    query_emb = mean_pooling(query_emb, encoded_query['attention_mask'])
    return cos_sim(query_emb[0], passage_emb[0]).tolist()

    
### bert words
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
bert_model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, 
                                  # We want to output all hidden states, as the second-to-last layer is of interest
                                  )

# Put the model in "evaluation" mode, meaning feed-forward operation.
bert_model.eval()

def bert_embedding(passage, idf):
    
    """
    Uses a pretrained Bert model to compute sentence embeddings for each sentence in the passage.
    Sentence embeddings are the weighted sum (in terms of idf) of contectualized word embeddings. 
    All sentences are fed simultaneously into Bert's forward pass.
    !The second-to-last hidden layer is defined as the word embedding layer! Alternatives are f.ex. taking the average of the 4 last hidden layers.
    
    Parameters
    ----------
    
    passage: String.
    
    Returns
    -------
    
    tensor of shape (number of sentences, 768)
    
    """
    sentences = sent_tokenize(passage) # list of sentences
    tokenized = tokenizer.__call__(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(tokenized["input_ids"], tokenized['token_type_ids'], tokenized["attention_mask"])
    
    hidden_states = outputs[2]
    sentence_embeddings = torch.empty(torch.Size((len(sentences), 768)))
    sentence_idf = get_idf(tokenized, idf)
    for i in range(len(sentences)):
        ### the sentence embedding is simply the weighted sum of the word embeddings. Weights are idf values, for simplicity.
        a = torch.full((768,len(sentence_idf[i])),1)
        b = torch.broadcast_tensors(sentence_idf[i], a)[0].transpose(1,0)
        sentence_embeddings[i] = torch.sum( #TODO! Does it matter that the weights do not add up to 1? Similarity not affected, so...
            torch.multiply(b,hidden_states[11][i])
            , dim = 0) # 11 is the second to last hidden layer, our embedding layer.
    return sentence_embeddings


def get_idf(tokenized, idf):
    """
    get idf value for every input token in the sentence(s).
    
    Parameters 
    ----------
    
    tokenized: transformers.tokenization_utils_base.BatchEncoding
            The tokenized passage or query.
    
    idf: dictionnary of form {"token": idf}
    
    Returns
    -------
    tensor of shape (number sentences, max length sentences)
    
    """
    
    number_sentences = tokenized["input_ids"].shape[0]
    out = torch.empty((number_sentences, len(tokenized["input_ids"][0])))
    for sentence_id in range(number_sentences):
        words = tokenizer.convert_ids_to_tokens(tokenized["input_ids"][sentence_id])

        for word_id in range(len(words)):
            if words[word_id] not in ['[PAD]', '[CLS]','[SEP]' ]: # to save lookup time for these tokens
                try:
                    out[sentence_id][word_id] = idf[preprocess(words[word_id])[0]] #we need to preprocess here, as the words in the dictionnary are preprocessed
                except:
                    out[sentence_id][word_id] = 0 # in case a word is not found in our term dictionnary.
                    #print("Word " + words[word_id] + " not found in term dictionnary")
            else:
                out[sentence_id][word_id] = 0
    return out

def similarity_bert_word_embeddings(query, passage, idf):
    """
    Using sentence embeddings as described in "bert_embeddings", computes the average cosine similarity 
    between the query and all the sentences in the passage.
    
    """
    query_embedding = bert_embedding(query, idf)
    passage_embedding = bert_embedding(passage, idf)
    if query_embedding.shape[0] > 1: # for the query, I just merge several sentences (should not occur very frequently)
        query_emb = torch.empty((1,768))
        for i in range(query_embedding.shape[0]):
            query_emb = query_emb.add(query_embedding[i])
        query_embedding = query_emb
        
    similarities = []
    for i in range(passage_embedding.shape[0]):
        similarities.append(cos_sim(query_embedding[0], passage_embedding[i]))
    s = similarities ### ??? If I delete this (useless) line, sometimes (f.ex. for obs. 590) na is outputted. ???
    return (sum(similarities)/passage_embedding.shape[0]).numpy()
  

if __name__ == "__main__":
    q = "What is the best ice cream?"
    p = "Without any doubt, chocolate ice cream is among the finest of all ice creams. It tastes delicious."
    print("query-passage similarity between")
    print(q)
    print(p+ " :")
    print("bm25: {}".format(bm25(idf, q, p)))
    print("bert_sentence_similarity: {}".format(cos_sim_sentence_embeddings(q, p)))
    print("bert_word_similarity.: {}".format(similarity_bert_word_embeddings(q, p, idf)))
    
