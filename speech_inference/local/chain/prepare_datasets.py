"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

import argparse
import os
import numpy as np

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--path_data', required=True)
ARGS = PARSER.parse_args()

path_data = ARGS.path_data
path_data_untrusted = ARGS.path_data[:-11]+'_untrusted_hires'
path_data_trusted = ARGS.path_data[:-11]+'_trusted_hires'

# Create a new directory if it does not alredy exist
if not os.path.exists(path_data_trusted):
    os.makedirs(path_data_trusted)
if not os.path.exists(path_data_untrusted):
    os.makedirs(path_data_untrusted)


def filter_ids(filename, ids):
    """
    Outputs a new file in the path_data_trusted with the same name as filename
    and with only the indices in ids
    """
    with open(os.path.join(path_data, filename), 'r') as f:
        content = f.readlines()
    output_f = open(os.path.join(path_data_trusted, filename), 'w')
    input_f = open(os.path.join(path_data_untrusted, filename), 'w')
    # Make sure to empty the files first in case.
    output_f.truncate(0)
    input_f.truncate(0)

    for line in content:
        if line.startswith(ids):
            output_f.write(line)
        else:
            input_f.write(line)


def filter_cmvn(filename, ids_trust, ids_no_trust):
    """
    Outputs two new files: one in the path_data_trusted with the same name as filename
    and with only the indices in ids_trust and one in the path_data_untrusted with the same name as filename
    and with only the indices in ids_no_trust
    """
    list_ids_trust = [id[: -5] for id in list(ids_trust)]
    ids_trust = tuple(i for i in list_ids_trust)

    list_ids_no_trust = [id[: -5] for id in list(ids_no_trust)]
    ids_no_trust = tuple(i for i in list_ids_no_trust)

    with open(os.path.join(path_data, filename), 'r') as f:
        content = f.readlines()

    outputfile = open(os.path.join(path_data_trusted, filename), 'w')
    inputfile = open(os.path.join(path_data_untrusted, filename), 'w')
    outputfile.truncate(0)
    inputfile.truncate(0)

    for line in content:
        if line.startswith(ids_trust):
            outputfile.write(line)
        if line.startswith(ids_no_trust):
            inputfile.write(line)


def get_list_ids(param):
    """
    Extracts ids outputted from the filtering module.
    """
    if not os.path.isfile('../codi/outputs/ids_{}.npy'.format(param)):
        print('Something went wrong in filtering. Exiting..')
        exit()
    index = np.load('../codi/outputs/ids_{}.npy'.format(param))
    index = index.T

    data = np.load('../codi/kaldi_outputs/dataset_unlabelled.npy', allow_pickle=True)
    # data_filtered = np.concatenate(data[index], axis=0)
    data_filtered = data[index]
    return ' '.join(data_filtered[:, 0].tolist())


IDS = get_list_ids(param='trust')
IDS_trust = tuple(map(str, IDS.split(' ')))
IDS = get_list_ids(param='no_trust')
IDS_no_trust = tuple(map(str, IDS.split(' ')))

for file in os.listdir(path_data):
    if any(file in s for s in ['wav.scp', 'utt2spk', 'text', 'feats.scp']):
        filter_ids(file, IDS_trust)
    if file == 'cmvn.scp':
        filter_cmvn(file, IDS_trust, IDS_no_trust)
