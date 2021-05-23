import sys
import statistics

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

MaxMRRRank = 10


def load_reference_from_stream(f):
    qids_to_relevant_passageids = {}
    for l in f:
        try:
            l = l.strip().split('\t')
            qid = int(l[0])
            if qid in qids_to_relevant_passageids:
                pass
            else:
                qids_to_relevant_passageids[qid] = []
            qids_to_relevant_passageids[qid].append(int(l[1]))
        except:
            raise IOError('\"%s\" is not valid format' % l)
    return qids_to_relevant_passageids


def load_reference(path_to_reference):
    with open(path_to_reference, 'r') as f:
        qids_to_relevant_passageids = load_reference_from_stream(f)
    return qids_to_relevant_passageids


def load_candidate_from_stream(f):
    qid_to_ranked_candidate_passages = {}
    for l in f:
        try:
            l = l.strip().split('\t')
            int(l[0].split('.')[0])
            qid = int(l[0].split('.')[0])
            pid = int(l[1].split('.')[0])
            rank = int(l[2].split('.')[0])
            if qid in qid_to_ranked_candidate_passages:
                pass
            else:
                # By default, all PIDs in the list of 1000 are 0. Only override those that are given
                tmp = [0] * 1000
                qid_to_ranked_candidate_passages[qid] = tmp
            qid_to_ranked_candidate_passages[qid][rank - 1] = pid
        except:
            raise IOError('\"%s\" is not valid format' % l)
    return qid_to_ranked_candidate_passages


def load_candidate(path_to_candidate):

    with open(path_to_candidate, 'r') as f:
        qid_to_ranked_candidate_passages = load_candidate_from_stream(f)
    return qid_to_ranked_candidate_passages


def quality_checks_qids(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    message = ''
    allowed = True

    # Create sets of the QIDs for the submitted and reference queries
    candidate_set = set(qids_to_ranked_candidate_passages.keys())
    ref_set = set(qids_to_relevant_passageids.keys())

    # Check that we do not have multiple passages per query
    for qid in qids_to_ranked_candidate_passages:
        # Remove all zeros from the candidates
        duplicate_pids = set(
            [item for item, count in Counter(qids_to_ranked_candidate_passages[qid]).items() if count > 1])

        if len(duplicate_pids - set([0])) > 0:
            message = "Cannot rank a passage multiple times for a single query. QID={qid}, PID={pid}".format(
                qid=qid, pid=list(duplicate_pids)[0])
            allowed = False

    return allowed, message


def compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    all_scores = {}
    MRR = 0
    ranking = []
    for qid in qids_to_ranked_candidate_passages:
        if qid in qids_to_relevant_passageids:
            ranking.append(0)
            target_pid = qids_to_relevant_passageids[qid]
            candidate_pid = qids_to_ranked_candidate_passages[qid]
            for i in range(0, MaxMRRRank):
                if candidate_pid[i] in target_pid:
                    print(i)
                    MRR += 1 / (i + 1)
                    ranking.pop()
                    ranking.append(i + 1)
                    break
    if len(ranking) == 0:
        raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")

    MRR = MRR * (1/len(qids_to_ranked_candidate_passages))
    all_scores['MRR @10'] = MRR
    all_scores['QueriesRanked'] = len(qids_to_ranked_candidate_passages)
    return all_scores


def compute_metrics_from_files(path_to_reference, path_to_candidate, perform_checks=True):
    qids_to_relevant_passageids = load_reference(path_to_reference)
    qids_to_ranked_candidate_passages = load_candidate(path_to_candidate)
    if perform_checks:
        allowed, message = quality_checks_qids(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)
        if message != '': print(message)
    return compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)


def main():

    path_to_reference = "/Users/kaibaeuerle/PycharmProjects/sentenceBert/qrels.train.tsv"
    #path_to_candidate = "/Users/kaibaeuerle/PycharmProjects/sentenceBert/final_result/all.csv"
    path_to_candidate = "/Users/kaibaeuerle/PycharmProjects/sentenceBert/final_result_cheat /all.csv"
    metrics = compute_metrics_from_files(path_to_reference, path_to_candidate)
    print('---------------------')
    for metric in sorted(metrics):
        print('{}: {}'.format(metric, metrics[metric]))
    print('---------------------')


if __name__ == '__main__':
    main()
