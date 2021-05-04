#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 17:44:12 2021

@author: Moritz
"""

import numpy as np
import nltk 
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#from spellchecker import SpellChecker # based on Peter Norvig's algorithm


### Idea: Wrap all the preprocessing steps in one function.
### Tokenization, SPelling correction, PorterStemmer, Punctuation removal and stopword removal are implemented.

ps = PorterStemmer()
stop_words = set(stopwords.words('english')).union(set([""])) 
#spell = SpellChecker()

def remove_punctuation(string):
    # initializing punctuations string
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    # Removing punctuations in string
    # Using loop + punctuation string
    for ele in string:
        if ele in punc:
            string = string.replace(ele, "")
    return string

def preprocess(sentence):
    """

    Parameters
    ----------
    sentence : String.

    Returns
    -------
    numpy array. Elements are tokens (Strings).

    """
    output = []

             
    for word in word_tokenize(sentence):
        #w = remove_punctuation(ps.stem(spell.correction(word)))
        w = remove_punctuation(ps.stem(word))
        if w not in stop_words:
            output.append(w)
    
    return np.array(output)


    