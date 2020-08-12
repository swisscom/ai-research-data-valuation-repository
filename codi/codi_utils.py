"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

import os
import matplotlib.pyplot as plt
import numpy as np
import Levenshtein
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

from features.features_classes.logits_feature import LogitsFeature
from features.features_classes.cost_feature import CostMargin
from features.features_classes.purity_feature import PurityFeats
from features.features_classes.parser import Parser


def generate_plots(values):
    """
    Generates plots of train and validation losses and accuracies
    """
    if not os.path.exists('codi/figures'):
        os.makedirs('codi/figures')
    plt.figure()
    for i in range(0, len(values)):
        plt.plot(values[i])
    plt.legend(('train_acc', 'train_loss', 'val_loss', 'val_acc'))
    plt.xlabel('# epochs')
    plt.ylabel('metric value')
    plt.savefig('codi/figures/metrics', format='pdf')


def plot_precision_threshold_tradeoff(pred, label):
    """
    Generates plots of precision of class 1 vs threshold of sorted points taken.
    """
    if not os.path.exists('codi/figures'):
        os.makedirs('codi/figures')
    print(classification_report(label, np.array(pred >= 0.5)))
    plt.figure()
    flat_preds = [item for sublist in pred.tolist() for item in sublist]
    pred_sorted = np.sort(np.array(flat_preds))[::-1]
    pred_sorted_idx = np.argsort(np.array(flat_preds))[::-1]
    true_sorted = label[pred_sorted_idx]
    precision_score = []
    for thresh in np.logspace(0, 2, 100)[:-1]:
        to_keep = np.floor((thresh * len(pred_sorted))/100)
        pred_artificial = np.array([pred_sorted >= pred_sorted[int(to_keep)]])
        precision_score.append(precision_recall_fscore_support(true_sorted,
                                                               pred_artificial[0])[0][1])
    plt.plot(np.logspace(0, 2, 100)[:-1], precision_score)
    plt.xlabel('Threshold')
    plt.ylabel('Precision of class 1')
    plt.title('Precision-Threshold tradeoff of class 1')
    plt.savefig('codi/figures/precision_threshold_tradeoff', format='pdf')


def filter_datapoints(test_preds, filtering):
    """
    Outputs an array of size (nb_thresh, ) of ids to retrain
    """
    test_preds = test_preds.squeeze()

    if not filtering['exp']:
        thresholds = filtering['thresh']
    else:
        if filtering['by_percentage'] is True:
            thresholds = np.logspace(1, 1.995, filtering['nb_thresh'])/100
            ordered_ids = np.argsort(test_preds)[::-1]
            ordered_preds = np.sort(test_preds)[::-1]
            N = len(ordered_preds[0:int(np.floor(thresholds[0]*len(ordered_preds)))])
            speakers = find_initial_speakers(ordered_ids[0:int(np.floor(thresholds[0]*len(ordered_preds)))])
        else:
            thresholds = 1.08-np.logspace(1, 1.995, filtering['nb_thresh'])/100
            N = len(np.where([test_preds >= thresholds[0]])[1])
            speakers = find_initial_speakers(np.where([test_preds >= thresholds[0]])[1])

    filtered_trust_data = []
    filtered_untrust_data = []

    if filtering['in_between'] is True:
        ids_keep = []
        ids_no_trust = []
        if filtering['by_percentage'] is True:
            thresholds = np.append([0], thresholds)
        else:
            thresholds = np.append([1], thresholds)
        prev_thresh = thresholds[0]

    for idx, thresh in enumerate(thresholds):
        if filtering['in_between'] is True:
            if idx != 0:
                if filtering['by_percentage'] is True:
                    ids = ordered_ids[int(np.floor(prev_thresh*len(ordered_preds))):
                                      int(np.floor(thresh*len(ordered_preds)))]

                else:
                    ids = np.where([(test_preds >= thresh) & (test_preds < prev_thresh)])[1]
                ids_keep = ids
                ids_no_trust = ids
                prev_thresh = thresh
        else:
            if filtering['by_percentage'] is True:
                ids_trust = ordered_ids[0:int(np.floor(thresh*len(ordered_preds)))]
                ids_no_trust = ordered_ids[int(np.floor(thresh*len(ordered_preds))):]
            else:
                ids_trust = np.where([test_preds >= thresh])[1]
                ids_no_trust = np.where([test_preds < thresh])[1]

            if filtering['exp'] == '1':
                # ids_keep = ids_trust[np.random.choice(len(ids_trust), N, replace=False)]
                ids_keep = filter_by_speaker(N, speakers, ids_trust)
            else:
                ids_keep = ids_trust

        if len(ids_keep) != 0:
            ids_keep = np.squeeze(ids_keep)
            filtered_trust_data.append(ids_keep)
        filtered_untrust_data.append(ids_no_trust)

    return [filtered_trust_data, filtered_untrust_data]


def find_initial_speakers(ids):
    """
    Finds the initial set of speakers (useful for filter_by_sepeaker).
    """
    X = np.load('codi/kaldi_outputs/dataset_unlabelled.npy', allow_pickle=True)
    X_keep = X[ids, :]
    speakers = []
    for j in range(0, len(X_keep)):
        speakers.append(X_keep[j, 0].split('-')[1])
    return set(speakers)


def filter_by_speaker(N, speakers, ids):
    """
    Instead of sampling randomly, we sample from the speakers present in the initial set of speakers.
    """
    X = np.load('codi/kaldi_outputs/dataset_unlabelled.npy', allow_pickle=True)
    X = X[ids, :]
    ids_keep = []
    speakers_in = []
    X, ids = shuffle(X, ids)
    for idx, x in enumerate(X):
        if ((x[0].split('-')[1] in speakers) and not (x[0].split('-')[1] in speakers_in)):
            speakers_in.append(x[0].split('-')[1])
            ids_keep.append(ids[idx])
    for idx, x in enumerate(X):
        if (len(ids_keep) < N):
            if ((ids[idx] not in ids_keep) and (x[0].split('-')[1] in speakers)):
                ids_keep.append(ids[idx])
    return ids_keep


def filter_nlp_datapoints(iterator, filtering):
    """
    Adds an attribute 'is_accepted' to the iterator as a boolean list of length nb_thresh.
    """
    if filtering['exp'] is False:
        thresholds = np.array(filtering['thresh'])
    else:
        if filtering['by_percentage'] is True:
            thresholds = np.linspace(10, 100, filtering['nb_thresh'])/100
        else:
            thresholds = 1.08 - np.logspace(1, 1.995, filtering['nb_thresh'])/100

    scores = np.ones((0, 1))

    for example_batch, _ in iterator:
        score_batch = [example.score for example in example_batch]
        scores = np.append(scores, score_batch, axis=0)

    if filtering['by_percentage'] is True:
        quantiles = np.quantile(scores, 1-thresholds)

    for example_batch, _ in iterator:
        for _, example in enumerate(example_batch):
            if filtering['in_between'] is False:
                if filtering['by_percentage'] is True:
                    example.is_accepted = quantiles <= example.score
                else:
                    example.is_accepted = thresholds <= example.score
            else:
                augmented_quantiles = np.append([1], quantiles)
                augmented_quantiles[-1] = 0
                prev_thresh = augmented_quantiles[0]
                for i in range(1, len(augmented_quantiles)):
                    if (example.score <= prev_thresh and example.score >= augmented_quantiles[i]):
                        example.is_accepted = [0]*filtering['nb_thresh']
                        example.is_accepted[i-1] = 1
                    prev_thresh = augmented_quantiles[i]


def save_ids(ids):
    """
    Saves IDS of points to retrain
    """
    if not os.path.exists('codi/outputs'):
        os.makedirs('codi/outputs')
    np.save('codi/outputs/ids_trust', ids[0])
    np.save('codi/outputs/ids_no_trust', ids[1])


def generate_box_plot(X, feature_dict, list_ids):
    """
    Plots boxplots of output of crossvalidation
    """
    if not os.path.exists('codi/figures'):
        os.makedirs('codi/figures')
    # Change list in dict to tuples.
    feature_dict = dict([a, tuple(x)] for a, x in feature_dict.items())
    # Invert values and keys in the dictionary
    inv_map = {v: k for k, v in feature_dict.items()}
    xticks_all = []
    plt.figure(figsize=[12, 7])

    for i in range(len(X.T)):
        xticks = []
        for j in range(0, len(list_ids[i])):
            if type(list_ids[i][j]) == list:
                xticks.append(inv_map[tuple(list_ids[i][j])])
            else:
                xticks.append(inv_map[list_ids[i][j]])
        xticks_all.append(xticks)
    plt.boxplot(X.T.tolist())
    plt.xlabel('Model features')
    plt.xticks(np.arange(1, len(X.T)+1), xticks_all, rotation=18, fontsize=5)
    plt.xlabel('Model features', fontsize=12)
    plt.ylabel('Precision of class 1')
    plt.savefig('codi/figures/precision_boxplot', format='pdf')


def plot_combination_accuracy(feature_dict, accs, list_ids, figname):
    """
    This function handles the plots of accuracy for each model after combination
    selection.
    """
    if not os.path.exists('codi/figures'):
        os.makedirs('codi/figures')
    # Change list in dict to tuples.
    feature_dict = dict([a, tuple(x)] for a, x in feature_dict.items())
    # Invert values and keys in the dictionary
    inv_map = {v: k for k, v in feature_dict.items()}
    xticks_all = []
    plt.figure(figsize=[12, 7])

    x = np.arange(0, len(accs))
    y = np.array(accs)

    for i in range(len(accs)):
        xticks = []
        for j in range(0, len(list_ids[i])):
            if type(list_ids[i][j]) == tuple:
                xticks.append(inv_map[list_ids[i][j]])
            else:
                xticks.append(inv_map[tuple(list_ids[i][j])])
        xticks_all.append(xticks)

    plt.plot(x, y)
    annot_max(x, y)
    plt.xlabel('Model features')
    plt.ylabel('{} accuracy'.format(figname))
    plt.xticks(np.arange(0, len(accs)), xticks_all, rotation=18, fontsize=5)
    plt.xlabel('Model features', fontsize=12)
    plt.savefig('codi/figures/{}'.format(figname), format='pdf')


def get_feature_dict(params):
    """
    Creates a dictionnary that maps the feature to their position in X
    """
    feature_dict = params['feature_dict']
    for key, value in feature_dict.items():
        if len(value) > 1:
            feature_dict[key] = tuple(range(value[0], value[1]))
    return feature_dict


def annot_max(x, y, ax=None):
    """
    Helper function to mark the maximum in a plot
    """
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text = 'x={:.3f}, y={:.3f}'.format(xmax, ymax)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=0.72)
    arrowprops = dict(arrowstyle='->', connectionstyle='angle,angleA=0,angleB=60')
    kw = dict(xycoords='data', textcoords='axes fraction',
              arrowprops=arrowprops, bbox=bbox_props, ha='right', va='top')
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.96), **kw)


def one_hot_encoder(y):
    """
    Creates one-hot-encoded version of target vector y for a binary
    classification task.
    """
    y_ohe = np.zeros((np.shape(y)[0], 2))
    y_ohe[y == 0, 0] = 1
    y_ohe[y == 1, 1] = 1
    return y_ohe


def merge_features(features):
    """
    Merges features into one X array
    """
    X = np.empty((np.shape(features[0])[1], 0))
    for feature in features:
        for feat in feature:
            if len(feat.shape) == 1:
                feat = np.expand_dims(feat, axis=1)
            X = np.concatenate([X, feat], axis=1)
    return X


# def merge_speech_features(features):
#     """
#     Merges features into one X array
#     """
#     X = np.empty((np.shape(features[0][0])[1], 0))
#     for feature in features:
#         for feat in feature:
#             if len(feat.shape) == 4:
#                 feat = feat.reshape((feat.shape[0], feat.shape[1], feat.shape[2]*feat.shape[3]))
#             elif len(feat.shape) == 2:
#                 feat = np.expand_dims(feat, axis=2)
#             elif len(feat.shape) == 1:
#                 feat = np.expand_dims(feat, axis=0)
#                 feat = np.expand_dims(feat, axis=2)
#             X = np.concatenate([X, feat[0]], axis=1)
#     return X


def merge_speech_features(feature):
    """
    Merges features into one X array
    """
    for idx, feat in enumerate(feature):
        if len(feat.shape) == 4:
            feat = feat.reshape((feat.shape[0], feat.shape[1], feat.shape[2]*feat.shape[3]))
        elif len(feat.shape) == 2:
            feat = np.expand_dims(feat, axis=2)
        elif len(feat.shape) == 1:
            feat = np.expand_dims(feat, axis=0)
            feat = np.expand_dims(feat, axis=2)
        if idx == 0:
            X = feat[0]
        else:
            X = np.concatenate([X, feat[0]], axis=1)
    return X


def create_speech_data(features, method, thresh):
    """
    Creates X and y arrays for the training, from kaldi's decoding output.
    """
    X_raw = np.load('codi/kaldi_outputs/dataset_labelled_2.npy', allow_pickle=True)
    if method == 'naive':
        y = X_raw[:, 8]
    elif method == 'levenshtein':
        if thresh is None:
            y = X_raw[:, 7]
        else:
            y = np.array(X_raw[:, 7] >= thresh).astype(int)

    parser = Parser()
    parser.parse(X_raw[:, 1])
    logits = parser.get_logits()
    logits_feats = LogitsFeature(sequence=True)
    costs_feats = CostMargin()
    purity_feats = PurityFeats()

    for idx, feature in enumerate(features):
        if feature == 'logits':
            logits_feats.augment(logits)
            feats_logits = logits_feats.get_features()
            if idx == 0:
                X = merge_speech_features(feats_logits)
            else:
                X = np.concatenate([X, merge_speech_features(feats_logits)], axis=1)

        if feature == 'nbests':
            costs_feats.augment(X_raw[:, 5], X_raw[:, 4])
            purity_feats.augment(X_raw[:, 6])
            nbest_feats = (costs_feats.get_features(), purity_feats.get_features())
            if idx == 0:
                X = merge_speech_features(nbest_feats)
            else:
                X = np.concatenate([X, merge_speech_features(nbest_feats)], axis=1)
    return X, y


def create_unlabelled_speech_data(features):
    """
    Creates X array from kaldi's decoding output.
    """
    X_raw = np.load('codi/kaldi_outputs/dataset_unlabelled.npy', allow_pickle=True)

    parser = Parser()
    parser.parse(X_raw[:, 1])
    logits = parser.get_logits()
    logits_feats = LogitsFeature(sequence=True)
    costs_feats = CostMargin()
    purity_feats = PurityFeats()

    for idx, feature in enumerate(features):
        if feature == 'logits':
            logits_feats.augment(logits)
            feats_logits = logits_feats.get_features()
            if idx == 0:
                X = merge_speech_features(feats_logits)
            else:
                X = np.concatenate([X, merge_speech_features(feats_logits)], axis=1)

        if feature == 'nbests':
            costs_feats.augment(X_raw[:, 4], X_raw[:, 3])
            purity_feats.augment(X_raw[:, 5])
            nbest_feats = (costs_feats.get_features(), purity_feats.get_features())
            if idx == 0:
                X = merge_speech_features(nbest_feats)
            else:
                X = np.concatenate([X, merge_speech_features(nbest_feats)], axis=1)
    return X


def get_speaker_distribution():
    """
    Displays informations about each experiment's set : number of different speakers and average levenshtein
    distance.
    """
    with open('codi/kaldi_outputs/text') as f:
        content = f.readlines()
    X = np.load('codi/kaldi_outputs/dataset_unlabelled.npy', allow_pickle=True)
    ids = np.load('codi/outputs/ids_trust.npy', allow_pickle=True)

    speakers_nb = []
    levenshtein_nb = []

    for i in range(0, len(ids)):
        speaker_id = []
        levenshtein = 0
        X_keep = X[ids[i], :]
        for j in range(0, len(X_keep)):
            # true_text = [s for s in content if s.startswith(X_keep[j, 0])][0].strip().split(' ', 1)[1]
            # true_text = [s for s in content if
            #              s.startswith(X_keep[j, 0].split('-', 1)[1])][0].strip().split(' ', 1)[1]
            true_text = [s for s in content if s.startswith(X_keep[j, 0][:-6])][0].strip().split(' ', 1)[1]
            # inferred_text = X_keep[j, 2].strip()
            inferred_text = X_keep[j, 2].strip().split(' ', 1)[1]
            levenshtein += float(Levenshtein.ratio(true_text, inferred_text))
            speaker_id.append(int(X_keep[j, 0].split('-')[0]))
        levenshtein_nb.append(levenshtein)
        speakers_nb.append(len(set(speaker_id)))
    thresholds = ['0.10', '0.16', '0.25', '0.40', '0.63', '0.99']
    levenshtein_nb_norm = [s/len(ids[0]) for s in levenshtein_nb]
    plt.figure()

    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.bar(np.arange(0, len(thresholds)), speakers_nb)
    ax2.plot(levenshtein_nb_norm, '-o', color='red')
    plt.xticks(np.arange(0, len(thresholds)), thresholds, fontsize=6.5)
    ax1.set_xlabel('Threshold used')
    ax1.set_ylabel('Count of different speakers', color='b')
    ax2.set_ylabel('Average Levenshtein distance', color='r')
    plt.show()
