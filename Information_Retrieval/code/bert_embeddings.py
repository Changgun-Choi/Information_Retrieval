# -*- coding: utf-8 -*-
"""
Created on Sun May  2 09:40:52 2021

@author: hp
"""

import torch
from transformers import BertTokenizer, BertModel
from nltk.tokenize import sent_tokenize
import sys
#sys.path.append('/media/Moritz/080FFDFF509A959E/BWsync_share/Master_BW/Information_Retrieval_Project/code')
sys.path.append("C:/Users/hp/bwSyncShare/Master_BW/Information_Retrieval_Project/code")
from preprocess import preprocess # own preprocessing function, own file

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, 
                                  # We want to output all hidden states, as the second-to-last layer is of interest
                                  )

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()
#%%
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
        outputs = model(tokenized["input_ids"], tokenized['token_type_ids'], tokenized["attention_mask"])
    
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
                
#%%

#test:
#idf = {"hello": 1, "friend": 4,"day": 0.01 }
#passage = "Hello my friend."
#passage = "Hello my friend. Day after day."
#sentences = sent_tokenize(passage) # list of sentences
#tokenized = tokenizer.__call__(sentences, padding=True, truncation=True, return_tensors="pt")          
#g = get_idf(tokenized, idf)

#a = torch.full((768,len(g[0])),1)
#b = torch.broadcast_tensors(g[0], a)[0].transpose(1,0)

#torch.multiply(b, hidden_states[11][0])


#bert = bert_embedding(passage, idf)
#bert.shape
#%%

def cos_sim(x,y):
    return torch.dot(x,y) / (torch.linalg.norm(x) * torch.linalg.norm(y))

def similarity_bert_word_embeddings(query, passage, idf):
    """
    Using sentence embeddings as described in "bert_embeddings", computes the average cosine similarity 
    between the query and all the sentences in the passage.
    
    """
    query_embedding = bert_embedding(query, idf)
    passage_embedding = bert_embedding(passage, idf)
    if query_embedding.shape[0] > 1:
        print("The query has more than one sentence!")
        return None
    else:
        similarities = []
        for i in range(passage_embedding.shape[0]):
            similarities.append(cos_sim(query_embedding[0], passage_embedding[i]))
        return sum(similarities)/passage_embedding.shape[0]
    
#%%
sim = similarity_bert_word_embeddings("How good is the coffee at Sammo?", "The coffee at Sammo is really outstanding. The students love to drink their coffee there.")
