"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

import argparse
import os
import numpy as np
import pandas as pd
import Levenshtein

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--path', required=True)
PARSER.add_argument('--unlabelled', required=True)
ARGS = PARSER.parse_args()


def load_file(path):
    """
    Returns a list of lines in file.
    """
    with open(path, 'r') as file:
        lines = file.readlines()
    return lines


def parse_probs(prob_string):
    """
    Returns the average difference between the two highest probabilities per word
    for each utterance
    """
    S = prob_string.strip().split('[')
    parsed_S = [''.join(s.strip().split(']')).strip().split(' ') for s in S]
    item_id = parsed_S[0][0]
    probs = []
    for prob_pairs in parsed_S[1:]:
        if len(prob_pairs) == 2:
            probs.append(float(prob_pairs[1]))
        elif len(prob_pairs) >= 4:
            probs.append(float(prob_pairs[1]) - float(prob_pairs[3]))

    avg = sum(probs)/len(probs)
    return item_id, avg


def find_true_label(x_id):
    """
    Returns the true label for each utterance
    """
    newpath = os.path.join('/workspace/jupyter/data/LibriSpeech/train-clean-100',
                           x_id.split('-')[0], x_id.split('-')[1])
    for filename in os.listdir(newpath):
        if filename.endswith('.txt'):
            x = load_file(os.path.join(newpath, filename))
    for line in x:
        temp_id = line.split()[0]
        if temp_id == x_id:
            return ''.join([i for i in line if not i.isdigit()]).strip('-').strip()


def parse_nbest(costs):
    """
    Returns the list of costs (LM, AC) for the nbest hypothesis
    """
    prev_id = costs[0].split(' ')[0][:-2]
    list_costs = []
    costs_dict = {}
    for i in range(0, len(costs)):
        if costs[i].split(' ')[0][:-2] == prev_id:
            costs_dict[costs[i].split(' ')[0][-1]] = costs[i].split(' ', 1)[1]
            if i == len(costs)-1:
                list_costs.append(costs_dict)
        else:
            list_costs.append(costs_dict)
            prev_id = costs[i].split(' ')[0][:-2]
            costs_dict = {}
            costs_dict[costs[i].split(' ')[0][-1]] = costs[i].split(' ', 1)[1]
    return list_costs


def tofloat(x):
    return x.astype(np.float128)


SAUSAGES = sorted(load_file(os.path.join(ARGS.path, 'mergedfile.txt')),
                  key=lambda a_line: a_line.split()[0])
PROB = load_file(os.path.join(ARGS.path, 'prob_avg.txt'))
PROBS = sorted([line for line in PROB if line.split(' ')[2].startswith('For')],
               key=lambda a_line: a_line.split(' ')[4])
TRANS = sorted(load_file(os.path.join(ARGS.path, 'decoded_text.txt')),
               key=lambda a_line: a_line.split()[0])
ACCS = sorted(load_file(os.path.join(ARGS.path, 'mergedac.txt')),
              key=lambda a_line: a_line.split()[0])
LMCS = sorted(load_file(os.path.join(ARGS.path, 'mergedlm.txt')),
              key=lambda a_line: a_line.split()[0])
NBEST = sorted(load_file(os.path.join(ARGS.path, 'nbest_text.txt')),
               key=lambda a_line: a_line.split()[0])

X_id = []
avg_prob = []
diff_prob = []
text = []

df = pd.DataFrame(columns=['X_id', 'sausages', 'avg_prob', 'diff_prob', 'text',
                           'lmcost', 'accost', 'nbest'])


for i in range(0, len(SAUSAGES)):
    item_id, avg = parse_probs(SAUSAGES[i])
    X_id.append(item_id)
    diff_prob.append(float(avg))
    avg_prob.append(float(PROBS[i].split('per-word')[1]))
    text.append(''.join([i for i in TRANS[i] if not i.isdigit()]).strip('-').strip())


df.X_id = X_id
df.avg_prob = avg_prob
df.diff_prob = diff_prob
df.text = text
df.lmcost = parse_nbest(LMCS)
df.accost = parse_nbest(ACCS)
df.nbest = parse_nbest(NBEST)
df.sausages = SAUSAGES

# Arrange format of acoustic cost and language model cost columns.
df['ac'] = df['accost'].apply(lambda x: np.array([a[1] for a in sorted([(int(k), float(v.strip()))
                                                                        for k, v in x.items()])]))
df['lm'] = df['lmcost'].apply(lambda x: np.array([a[1] for a in sorted([(int(k), float(v.strip()))
                                                                        for k, v in x.items()])]))

df['ac'] = df['ac'].apply(tofloat)
df['lm'] = df['lm'].apply(tofloat)

if not os.path.exists('codi/kaldi_outputs'):
    os.makedirs('codi/kaldi_outputs')

if ARGS.unlabelled == 'no':
    true_label = []
    for i in range(0, len(SAUSAGES)):
        item_id = X_id[i]
        true_label.append(find_true_label(item_id))
    df['true_text'] = true_label
    df['levenstein'] = None
    for idx, row in df.iterrows():
        df['levenstein'][idx] = float(Levenshtein.ratio(row['true_text'], row['text']))
    df['label'] = list(df.text == df.true_text)
    df['label'] = df['label'].astype(int)
    X = df[['X_id', 'sausages', 'text', 'true_text', 'lm', 'ac', 'nbest', 'levenstein', 'label']].to_numpy()
    np.save('codi/kaldi_outputs/dataset_labelled_2', X)
else:
    X = df[['X_id', 'sausages', 'text', 'lm', 'ac', 'nbest']].to_numpy()
    np.save('codi/kaldi_outputs/dataset_unlabelled', X)
