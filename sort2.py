import pandas as pd
import datetime
import csv

from sentence_transformers import SentenceTransformer
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np

queries = pd.read_csv(
    "/Users/kaibaeuerle/PycharmProjects/sentenceBert/queries.train.tsv",
    sep=",", header=None)
queries = queries.to_numpy()
passage = pd.read_csv(
    "/Users/kaibaeuerle/PycharmProjects/sentenceBert/collection.tsv",
    sep=",", header=None)
passage = passage.to_numpy()
data = pd.read_csv(
    "/Users/kaibaeuerle/PycharmProjects/sentenceBert/all.csv",
    sep=",", header=None)
data = data.to_numpy()
unqiue_ids = np.unique(data[:, 0])
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
    return elem[2]


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def from_data_to_array(data):
    data_ma = []
    for row in data:
        data_ma.append([remove_punctuation(queries[queries[:, 0] == row[0], 1][0].lower()),
                        remove_punctuation(passage[passage[:, 0] == row[1], 1][0].lower())])
    return data_ma


for id in unqiue_ids:
    a = datetime.datetime.now()
    matrix = []
    data_ma = from_data_to_array(data[data[:, 0] == id])
    for i in range(0, len(data_ma)):
        matrix.append(
            [data_ma[i, 0], data_ma[i, 1], cosine(sbert_model.encode([data_ma[i][0]])[0], sbert_model.encode(data_ma[i][1]))])
    matrix = sorted(matrix, key=take_third, reverse=True)
    with open("/Users/kaibaeuerle/PycharmProjects/sentenceBert/result/" + str(id) + ".csv", 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')
        mywriter.writerows(matrix)
    print("ID: ", id, " Time: ", datetime.datetime.now() - a)
