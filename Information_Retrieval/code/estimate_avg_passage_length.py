#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 19:07:21 2021

@author: Moritz
"""

import pandas as pd
import sys
import numpy as np
sys.path.append('/media/Moritz/080FFDFF509A959E/BWsync_share/Master_BW/Information_Retrieval_Project/code')
from preprocess import preprocess # own preprocessing function

path = "/media/Moritz/Seagate Backup Plus Drive/University/Information_Retrieval_Project/data/"


### estimate average passage length (after preprocessing).
### Passages are drawn AT RANDOM, to avoid bias if first passages are shorter than last passages 
### in the collection...
sample_size = 50000
passages = pd.read_csv(path + "collection.tsv",
                    sep = "\t",  header = None)
indexes = np.random.randint(0,8841823, sample_size)
passages_bow = np.array(passages[1][indexes].apply(preprocess))

count = 0
for p in passages_bow:
    count += len(p)
    
print(count/sample_size)
# 35.4