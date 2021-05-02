# -*- coding: utf-8 -*-
"""
Created on Sun May  2 09:40:52 2021

@author: hp
"""

import torch
from transformers import BertTokenizer, BertModel
from nltk.tokenize import sent_tokenize

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()
#%%
passage = "We met at the bar down by the river. It was a cold evening! However, we were all well dressed."
def bert_embedding(passage):
    
    """
    Uses a pretrained Bert model to compute sentence embeddings for each sentence in the passage.
    Sentence embeddings are simply the average of the contextualized word embeddings. 
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

        outputs = model(tokenized["input_ids"], tokenized['token_type_ids'], tokenized["attention_mask"])
    hidden_states = outputs[2]
    sentence_embeddings = torch.empty(torch.Size((len(sentences), 768)))
    for i in range(len(sentences)):
        sentence_embeddings[i] = torch.sum(hidden_states[11][i], dim = 0) # 11 is the second to last hidden layer, our embedding layer.
    return sentence_embeddings


   
#%%
## test implementation:
e = bert_embedding(passage) 

#%%

def cos_sim(x,y):
    return torch.dot(x,y) / (torch.linalg.norm(x) * torch.linalg.norm(y))

def similarity(query, passage):
    """
    Using sentence embeddings as described in "bert_embeddings", computes the average cosine similarity 
    between the query and all the sentences in the passage.
    
    """
    query_embedding = bert_embedding(query)
    passage_embedding = bert_embedding(passage)
    if query_embedding.shape[0] > 1:
        print("The query has more than one sentence!")
        return None
    else:
        similarities = []
        for i in range(passage_embedding.shape[0]):
            similarities.append(cos_sim(query_embedding[0], passage_embedding[i]))
        return sum(similarities)/passage_embedding.shape[0]
    
#%%
sim = similarity("How great is the bass guitar?", "In the 20th century, the electric bass guitar gained a lot of popularity. It rocks!")
