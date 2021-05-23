import csv

import pandas as pd
from pathlib import Path
import os, glob
import pandas as pd

def ranking():
    #pathlist = Path("/Users/kaibaeuerle/PycharmProjects/sentenceBert/result").rglob('*.csv')
    pathlist = Path("/Users/kaibaeuerle/PycharmProjects/sentenceBert/result_cheat").rglob('*.csv')
    for path in pathlist:
        result = pd.read_csv(
            str(path),
            sep=",", header=None)
        result = result.to_numpy()

        for i, element in enumerate(result):
            result[i, 2] = i + 1
            result[i, 0] = int(result[i, 0])
            result[i, 1] = int(result[i, 1])
            result[i, 2] = int(result[i, 2])
        #with open("/Users/kaibaeuerle/PycharmProjects/sentenceBert/final_result/" + str(path).split('/')[6], 'w', newline='') as file:
        with open("/Users/kaibaeuerle/PycharmProjects/sentenceBert/final_result_cheat /" + str(path).split('/')[6], 'w', newline='') as file:
            mywriter = csv.writer(file, delimiter='\t')
            mywriter.writerows(result)

def combining():
    #path = r'/Users/kaibaeuerle/PycharmProjects/sentenceBert/final_result'  # use your path
    path = r'/Users/kaibaeuerle/PycharmProjects/sentenceBert/final_result_cheat '  # use your path
    all_files = glob.glob(path + "/*.csv")

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=None, sep="\t")
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame = frame.to_numpy()
    #with open("/Users/kaibaeuerle/PycharmProjects/sentenceBert/final_result/all.csv", 'w', newline='') as file:
    with open("/Users/kaibaeuerle/PycharmProjects/sentenceBert/final_result_cheat /all.csv", 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter='\t')
        mywriter.writerows(frame)

ranking()
combining()