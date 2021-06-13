# -*- coding: utf-8 -*-
"""
Created on Sun May  9 11:01:32 2021

@author: Moritz
"""

import pandas as pd
import numpy as np
from sklearn import svm
import sklearn as sk
import sklearn.ensemble as ske
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
import pickle
import statsmodels.api as sm # better summary table than sklearn's version
from sklearn.metrics import roc_auc_score, accuracy_score

path = "/media/Moritz/080FFDFF509A959E/MXLinux/Information_Retrieval_Project/data/"

def save_obj(obj, name ):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

df = pd.read_csv(path + "training_20000.csv")


X = df[["bm25","bert_sim","bert_sentence_sim"]]
scaler = sk.preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
X = sm.add_constant(X)
save_obj(scaler, "scaler")

y = df["relevant"]

# for interaction effects:
X_inter = df[["bm25","bert_sim","bert_sentence_sim"]].copy()
X_inter["bm25_bert_sim"] = X_inter["bm25"] * X_inter["bert_sim"]
X_inter["bm25_bert_sentence_sim"] = X_inter["bm25"] * X_inter["bert_sentence_sim"]
X_inter["bert_sim_bert_sentence_sim"] = X_inter["bert_sim"] * X_inter["bert_sentence_sim"]
X_inter["all"] = X_inter["bert_sim"] * X_inter["bm25"] * X_inter["bert_sentence_sim"]
scaler2 = sk.preprocessing.StandardScaler().fit(X_inter)
X_inter = scaler2.transform(X_inter) # Interaction attributes are scaled after multiplication.
X_inter = sm.add_constant(X_inter)
#%% Discriminatory power and correlation between the single variables
print("Covariance matrix of the variables:")
print("")
print(np.cov(X[:,1:4], rowvar = False)) # only mild correlations! Combining the variables might lead to better results.
print("")
print("")
bm25_model = sk.linear_model.LogisticRegression(penalty = "none")
bm25_model.fit(X.transpose()[1].reshape(-1, 1), y)
print("auc bm25:")
print(sk.metrics.roc_auc_score(y, bm25_model.predict(X.transpose()[1].reshape(-1, 1))))
print("")
bert_words_model = sk.linear_model.LogisticRegression(penalty = "none")
bert_words_model.fit(X.transpose()[2].reshape(-1, 1), y)
print("auc bert_words:")
print(sk.metrics.roc_auc_score(y, bert_words_model.predict(X.transpose()[2].reshape(-1, 1))))
print("")
bert_sent_model = sk.linear_model.LogisticRegression(penalty = "none")
bert_sent_model.fit(X.transpose()[3].reshape(-1, 1), y)
print("auc bert_sent:")
print(sk.metrics.roc_auc_score(y, bert_sent_model.predict(X.transpose()[3].reshape(-1, 1))))
print("")


#%% logistic regression
"""
logistic = sk.linear_model.LogisticRegression(penalty = "none") # with 3 variables, a penalty term is not really needed

accuracy = sk.model_selection.cross_val_score(logistic, X, y, cv=5, scoring = "accuracy")
print(accuracy)

auc = sk.model_selection.cross_val_score(logistic, X, y, cv = 5, scoring="roc_auc")
print(auc)
"""

df = pd.DataFrame(X, columns = "constant bm25 bert_sim bert_sentence_sim".split(" "))
logistic = sm.Logit(y, df).fit()
print(logistic.summary()) 
with open(path + "logistic_model.pkl", 'wb') as f:
    pickle.dump(logistic, f)

preds = logistic.predict(df)
print("")
print("AUC on entire training set:")
print(roc_auc_score(y, preds))
print("")
print(logistic.summary().as_latex())

"""
#%% Adding interaction effects:


logistic = sk.linear_model.LogisticRegression() # here, a penalty term is used

print("Accuracy with possible interaction effects:")
accuracy = sk.model_selection.cross_val_score(logistic, X_inter, y, cv=5, scoring = "accuracy")
print(accuracy)
print("")
print("AUC with possible interaction effects")
auc = sk.model_selection.cross_val_score(logistic, X_inter, y, cv = 5, scoring="roc_auc")
print(auc)
print("")
print("Variable importance:") 
#print(logistic.get_params()) Need to update scikit learn... currently no space left on device

logistic = sm.Logit(y, X_inter).fit()
print(logistic.summary())

logistic = sm.Logit(y, X_inter[:,[0,5]]).fit()
print(logistic.summary())

logistic = sk.linear_model.LogisticRegression(penalty = "none") # with 2 variables, a penalty term is not really needed
accuracy = sk.model_selection.cross_val_score(logistic, X_inter[:,[0,5]], y, cv=5, scoring = "accuracy")
print(accuracy)

auc = sk.model_selection.cross_val_score(logistic, X_inter[:,[0,5]], y, cv = 5, scoring="roc_auc")
print(auc)

"""

df_inter2 = X_inter[:,0:7]
logistic2 = sm.Logit(y, df_inter2).fit()
print(logistic2.summary())

save_obj(logistic2, "logistic_model_interaction")

# save scaler2 seperately, so that it only scales the two variables:
df_inter2_scaled = df[["bm25"]].copy()
df_inter2_scaled["bert"] = df["bert_sim"] * df["bert_sentence_sim"]

scaler2 = sk.preprocessing.StandardScaler().fit(df_inter2_scaled)
save_obj(scaler2, "scaler_interaction")


"""
#%% nonlinear SVM
# Brute Force Grid Search is used to estimate good values for gamma and C
# (Code copied directly from sklearn docs)

C_range = [1.5,3,10]
gamma_range = [0.01,0.1,0.5]
param_grid = dict(gamma=gamma_range, C=C_range)
cv = sk.model_selection.StratifiedShuffleSplit(n_splits=4, test_size=0.2, random_state=123)
grid = sk.model_selection.GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, y)
grid.best_params_
sk.metrics.roc_auc_score(y, grid.predict(X))




C_range = np.logspace(-2, 4, 13)
gamma_range = np.logspace(-3, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
base_estimator = svm.SVC(random_state=123)
sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
                          factor=2).fit(X, y)
sh.best_estimator_
# C = 10000, gamma = 0.0031622776601683794

svmc = svm.SVC(C = 10000, gamma = 0.00316, random_state = 123)
svmc.fit(X,y)
sk.metrics.roc_auc_score(y, svmc.predict(X))
accuracy = sk.model_selection.cross_val_score(svmc, X, y, cv=5, scoring = "accuracy")
print(accuracy)
auc = sk.model_selection.cross_val_score(svmc, X, y, cv = 5, scoring="roc_auc")
print(auc)
#%% Random Forest Classifier

rf = ske.RandomForestClassifier()
accuracy = sk.model_selection.cross_val_score(rf, X, y, cv=5, scoring = "accuracy")
print(accuracy)

auc = sk.model_selection.cross_val_score(rf, X, y, cv = 5, scoring="roc_auc")
print(auc)

#%%

param_grid = {'max_depth': [3, 5, 10, 25],
               'min_samples_split': [2, 5, 10]}
base_estimator = RandomForestClassifier(random_state=123)
sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
                          factor=2, resource='n_estimators',
                          max_resources=300).fit(X, y)
sh.best_estimator_
"""




