# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 12:07:01 2021

@author: hp
"""
import os

import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download()


def data_import():
    # %% Read data
    path = "/Users/kaibauerle/Desktop/Uni Mannheim/Module/Information Retrieval and Web Search/Project/"
    # path = "F:/University/Information_Retrieval_Project/data/"

    # to concatenate positive and negative examples ("split up" the data set)
    # read positive and then negative entries
    train_pos = pd.read_csv(path + "triples.train.small.tsv",
                            sep="\t", nrows=100, header=None, skiprows=1, usecols=[0, 1])
    train_neg = pd.read_csv(path + "triples.train.small.tsv",
                            sep="\t", nrows=100, header=None, skiprows=1, usecols=[0, 2])

    # %% Data pre-processing

    train_pos.columns = ["query", "passage"]
    train_neg.columns = ["query", "passage"]
    train_pos["relevant"] = 1  # target label
    train_neg["relevant"] = 0

    train = train_pos.append(train_neg)

    # %% Feature computation

    X = train.copy()

    X["number_chars_q"] = X["query"].apply(count_chars)
    X["number_chars_p"] = X["passage"].apply(count_chars)

    # preprocseeing
    X_queries = X.to_numpy()[:,0]
    X_passages = X.to_numpy()[:,1]

    # lower casing

    # stemming
    for query in range(len(X_queries)):
        X_queries[query] = stemSentence(X_queries[query])
        X_queries[query] = remove_punctoation(X_queries[query])
    for passage in range(len(X_passages)):
        X_passages[passage] = stemSentence(X_passages[passage])
        X_passages[passage] = remove_punctoation(X_passages[passage])
    # TFIDF & stopwords
    vectorizer = TfidfVectorizer(stop_words='english')
    X_queries_preprocessed = vectorizer.fit_transform(X_queries)
    X_passages_preprocessed = vectorizer.fit_transform(X_passages)
    X_all_pre = vectorizer.fit_transform(X_queries+X_passages)

    print(X_all_pre.shape)
    print(X)
    print(X_queries[0])
    print(X_queries_preprocessed[0])
    print(X_passages[0])
    print(X_passages_preprocessed[0])


def stemSentence(sentence):
    """
    this method stems the sentences
    :param sentence: sentences to be stemmed
    :return: the stemmed sentences
    """
    # create an object of PorterStemmer
    porter = PorterStemmer()
    token_words = word_tokenize(sentence)
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def remove_punctoation(string):
    # initializing punctuations string
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    # Removing punctuations in string
    # Using loop + punctuation string
    for ele in string:
        if ele in punc:
            string = string.replace(ele, "")
    return string

def count_chars(x):
    return len(x)
