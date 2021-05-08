import csv

from sentence_transformers import SentenceTransformer
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')


def remove_punctuation(string):
    # initializing punctuations string
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    # Removing punctuations in string
    # Using loop + punctuation string
    for ele in string:
        if ele in punc:
            string = string.replace(ele, "")
    return string


def take_third(elem):
    return elem[0]


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


path = "/Users/kaibauerle/Desktop/Uni Mannheim/Module/Information Retrieval and Web Search/Project/"
train_pos = pd.read_csv(path + "triples.train.small.tsv",
                        sep="\t", nrows=10000, header=None, skiprows=1, usecols=[0, 1])
train_neg = pd.read_csv(path + "triples.train.small.tsv",
                        sep="\t", nrows=10000, header=None, skiprows=1, usecols=[0, 2])
train_pos = train_pos.to_numpy()
train_neg = train_neg.to_numpy()
train_sent = []
for s in train_pos:
    train_sent.append(s[1].lower())
for s in train_neg:
    train_sent.append(s[1].lower())
sentence_embeddings = sbert_model.encode(train_sent)
train_all = np.concatenate((train_pos, train_neg))
query = remove_punctuation(train_pos[0][0].lower())
query_vec = sbert_model.encode([query])[0]

save_array = []
for i in range(0, len(train_all)):
    sim = cosine(sbert_model.encode([train_all[i][0]])[0], sbert_model.encode([train_all[i][1]])[0])
    # print("Sequence = ", train_all[i][0], "; Passage = ", train_all[i][1], "; similarity = ", sim)
    save_array.append([train_all[i][0], train_all[i][1], sim])
# save_array.sort(key=take_third)
count_right = 0
for i in range(0, 9999):
    if save_array[i][2] > save_array[i + 10000][2]:
        print("yes", save_array[i][2], save_array[i + 10000][2])
        count_right = count_right + 1
    else:
        print("no", save_array[i][2], save_array[i + 10000][2])
print("Number of rights: ", count_right)
with open('myfile.csv', 'w', newline='') as file:
    mywriter = csv.writer(file, delimiter=',')
    mywriter.writerows(save_array)