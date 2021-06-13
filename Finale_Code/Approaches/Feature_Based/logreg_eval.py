# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:11:05 2021

@author: Moritz
"""

import pickle 
import numpy as np
import time
from multiprocessing import Pool
import statsmodels.api as sm
#path_data = "E:/University/Information_Retrieval_Project/data/eval_data_features/"
#path = "D:/MXLinux/Information_Retrieval_Project/data/"
path = "/media/Moritz/080FFDFF509A959E/MXLinux/Information_Retrieval_Project/data/"
path_data = "/media/Moritz/Seagate Backup Plus Drive/University/Information_Retrieval_Project/data/eval_data_features/"
def load_obj(name):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)
def load_eval_data(name):
    with open(path_data + name + '.pkl', 'rb') as f:
        return pickle.load(f)   

    


### load trained models

logreg = load_obj("logistic_model")
scaler = load_obj("scaler")
logreg2 = load_obj("logistic_model_interaction")
scaler2 = load_obj("scaler_interaction")

logreg_pairwise = load_obj("logreg_pairwise")
scaler_pairwise = load_obj("scaler_pairwise")
### load evaluation data
r = range(200) 
for chunk_index in r:
    if chunk_index == list(r)[0]:     
        df = load_eval_data("all_features_{}".format(chunk_index * 1000))
    else:
        df = df.append(load_eval_data("all_features_{}".format(chunk_index * 1000)), ignore_index = True)
        

qids = df["qid"].unique()

### preprocess data and apply model
X = df.loc[:,["bm25","bert_sim_word","bert_sim_sentence"]]
X = scaler.transform(X)
X = sm.add_constant(X)
probas = logreg.predict(X)



df["proba"] = probas



                    
#%%

def rank(qids):

    i = 1
    for q in qids:
        
        start = time.time()    
        
        # save pointwise ranking
        
        df_pointwise = df[df["qid"] == q].sort_values(by = "proba", ascending = False)
        df_pointwise["ranking"] = list(range(1,len(df_pointwise) + 1))
        #df_pointwise[["qid","pid","ranking"]].to_csv("E:/University/Information_Retrieval_Project/data/eval_data_for_kai/logreg/logreg_{}.csv".format(q), index = False)
        df_pointwise[["qid","pid","ranking"]].to_csv("/media/Moritz/Seagate Backup Plus Drive/University/Information_Retrieval_Project/data/eval_data_for_kai/logreg/logreg_{}.csv".format(q), index = False)
        
        """
        # save pointwise ranking based on model with interaction effect
        df_pointwise = df[df["qid"] == q].sort_values(by = "proba_inter", ascending = False)
        df_pointwise["ranking"] = list(range(1,len(df_pointwise) + 1))
        #df_pointwise[["qid","pid","ranking"]].to_csv("E:/University/Information_Retrieval_Project/data/eval_data_for_kai/logreg_intersection/logreg_intersection_{}.csv".format(q), index = False)
        df_pointwise[["qid","pid","ranking"]].to_csv("/media/Moritz/Seagate Backup Plus Drive/University/Information_Retrieval_Project/data/eval_data_for_kai/logreg_intersection/logreg_intersection_{}.csv".format(q), index = False)
        """
        
        # now perform pairwise reranking. 
        df_pairwise = df.loc[df["qid"] == q].copy().sort_values(by = "proba", ascending = False)
        score_pairwise_sum = [] # one score for each passage
        score_pairwise_binary = []
        for p1 in df_pairwise.iloc[0:50].loc[:,"pid"]:
            probas_pairwise = [] # one proba for each p1 - p2 combination (p1 fixed)
            for p2 in df_pairwise.iloc[0:50].loc[:,"pid"]:
                if p1 == p2:
                    pass # only compare to the k-1 different passages
                else:
                    # derive features from the features for the pointwise model
                    features = (df_pairwise[df_pairwise["pid"] == p1][["bm25","bert_sim_word","bert_sim_sentence"]].values
                         - df_pairwise[df_pairwise["pid"] == p2][["bm25","bert_sim_word","bert_sim_sentence"]].values) 
                    features = scaler_pairwise.transform(features)
                    # add constant:
                    features = np.concatenate([[1],features[0]])
                    # apply model - get proba that p1 is more relevant than p2
                    probas_pairwise.append(logreg_pairwise.predict(features))
                 
            # Here, the sum is chosen as a means to get a score for passage p1
            score_pairwise_sum.append(sum(probas_pairwise))
            # Here, the binary score
            score_pairwise_binary.append(sum(np.array(probas_pairwise) > 0.5))
        
        df_pairwise["pairwise_sum"] = 0
        df_pairwise["pairwise_binary"] = 0
        df_pairwise.iloc[0:50,df_pairwise.columns == "pairwise_sum"] = score_pairwise_sum
        df_pairwise.iloc[0:50,df_pairwise.columns =="pairwise_binary"] = score_pairwise_binary
        
        # important: do not sort the entire dataframe based on the columns "pairwise_sum/binary", 
        # only the top 50 so that the pointwise ordering for the rest of the observations remains unchanged.
        df_pairwise.iloc[0:50] = df_pairwise.iloc[0:50].sort_values(by = "pairwise_sum", ascending = False)
        df_pairwise[["ranking"]] = list(range(1,len(df_pairwise) + 1))
        df_pairwise[["qid","pid","ranking"]].to_csv("/media/Moritz/Seagate Backup Plus Drive/University/Information_Retrieval_Project/data/eval_data_for_kai/pairwise_logreg_sum/pairwise_logreg_sum{}.csv".format(q), index = False)
        
        df_pairwise.iloc[0:50] = df_pairwise.iloc[0:50].sort_values(by = "pairwise_binary", ascending = False)
        df_pairwise[["ranking"]] = list(range(1,len(df_pairwise) + 1))
        df_pairwise[["qid","pid","ranking"]].to_csv("/media/Moritz/Seagate Backup Plus Drive/University/Information_Retrieval_Project/data/eval_data_for_kai/pairwise_logreg_binary/pairwise_logreg_binary{}.csv".format(q), index = False)
        
        print("{}: {}".format(i, time.time() - start))   
        i += 1              


#%%
qids_all = df["qid"].unique() # length: 203
qids_all = [qids_all[0:33],qids_all[33:67], qids_all[67:100],qids_all[100:134],qids_all[134:168], qids_all[168:203]]

def main():
    
    print('starting computations on 6 cores')
    with Pool() as pool:
        pool.map(rank, qids_all)
    
    



if __name__ == '__main__':
    main()   
                  
                       
            
                     
                     
                  
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                
                
                
                
                
                
                
                
                
                
