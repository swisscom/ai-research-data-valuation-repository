"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

import torch
import numpy as np
from features.features_classes.logits_feature import LogitsFeature
import copy
from scipy.stats import entropy
from sklearn.metrics import f1_score


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch
    :param preds: prediction logits
    :param y: target labels
    :return: accuracy = percentage of correct predictions
    """

    # round predictions to the closest integer
    rounded_predictions = torch.round(torch.sigmoid(preds))
    correct = (rounded_predictions == y).float()
    acc = correct.sum() / len(correct)

    return acc


def binary_f1_score(preds, y):
    """
    Returns F1-score per batch
    :param preds: prediction logits
    :param y: target labels
    :return: score = F1-score
    """
    rounded_predictions = torch.round(torch.sigmoid(preds))

    return f1_score(y.cpu().numpy(), rounded_predictions.cpu().numpy(), average='weighted')


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    :param preds: prediction logits
    :param y: target labels
    :return: categorical accuracy
    """
    max_preds = preds.argmax(dim=1, keepdim=True)
    correct = max_preds.squeeze(1).eq(y)

    return correct.sum() / torch.FloatTensor([y.shape[0]])


def categorical_f1_score(preds, y):
    max_preds = preds.argmax(dim=1, keepdim=True).squeeze(1)

    return f1_score(y.cpu().numpy(), max_preds.cpu().numpy(), average='weighted')


def train(model, iterator, optimizer, criterion, binary=True):
    """
    Train a PyTorch model
    :param model: the PyTorch model
    :param iterator: dataset in batch iterator form
    :param optimizer: optimizer for the training
    :param criterion: criterion between predictions and target
    :param binary: whether we work with binary classes or multi-classes
    :return: mean loss and accuracy
    """
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        if binary:
            predictions = model(batch.text).squeeze(1)
        else:
            predictions = model(batch.text)

        loss = criterion(predictions, batch.label)

        if binary:
            acc = binary_accuracy(predictions, batch.label)
        else:
            acc = categorical_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, binary=True):
    """
    Evaluate a PyTorch model given a dataset (in batch iterator form) and a criterion
    :param model: the PyTorch model
    :param iterator: iterator over batches of data
    :param criterion: criterion to be used to compare target and predictions
    :param binary: whether we work with binary classes or multi-classes
    :return: mean loss and accuracy over the provided dataset
    """
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0

    # Put the model in eval mode
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            if binary:
                predictions = model(batch.text).squeeze(1)
            else:
                predictions = model(batch.text)

            loss = criterion(predictions, batch.label)

            if binary:
                acc = binary_accuracy(predictions, batch.label)
                f1 = binary_f1_score(predictions, batch.label)
            else:
                acc = categorical_accuracy(predictions, batch.label)
                f1 = categorical_f1_score(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_f1 += f1.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_f1 / len(iterator)


def predict(model, iterator, binary=True):
    """
    Predicts logits and labels with a trained model and outputs the original labels as well.
    All operations are done in-place on the dataset itself.
    :param model: the PyTorch model
    :param iterator: iterator over batches of data
    :param binary: whether we work with binary classes or multi-classes
    """
    # Put the model in eval mode
    model.eval()

    with torch.no_grad():
        for example_batch, batch in iterator:
            if binary:
                predictions_torch = model(batch.text).squeeze(1)
                labels_torch = torch.round(torch.sigmoid(predictions_torch))
            else:
                predictions_torch = model(batch.text)
                labels_torch = predictions_torch.argmax(dim=1, keepdim=True).squeeze(1)

            for index, example in enumerate(example_batch):
                # Retrieve the corresponding label
                example.predicted_label = iterator.dataset.fields['label'].vocab.itos[labels_torch[index]]

                if example.predicted_label == example.label:
                    example.is_correct = True
                else:
                    example.is_correct = False

                if binary:
                    example.logit = double_logits(predictions_torch[index].sigmoid().detach().cpu().numpy())
                else:
                    temp_logit = predictions_torch[index].detach().cpu().numpy()

                    # We avoid negative values for entropy computation
                    example.logit = (temp_logit + np.abs(temp_logit))/2

                logits_features = LogitsFeature()
                logits_features.augment(np.expand_dims(example.logit, axis=0))
                _, example.margin, example.ratio, example.entropy = logits_features.get_features()


def epoch_time(start_time, end_time):
    """
    Computes the time for each epoch in minutes and seconds.
    :param start_time: start of the epoch
    :param end_time: end of the epoch
    :return: time in minutes and seconds
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


def double_logits(input_logits):
    """
    Double input logits.
    Doubling an input logits of shape (n, 1) turns it into a logits of shape (n, 2) following one-hot fashion.
    :param input_logits: logits of shape (n, 1)
    :return: logits of shape (n, 2)
    """
    if len(input_logits.shape) == 0:
        value_logit = float(input_logits)
        return np.array([1 - value_logit, value_logit])

    input_shape = input_logits.shape
    twin_logits = np.ones(input_shape) - input_logits

    output_logits = np.stack((twin_logits, input_logits), axis=1)

    return output_logits


def compute_percent_correct(array_of_examples):
    is_correct_list = [example.is_correct for example in array_of_examples]

    return np.mean(is_correct_list)


def prediction_example(example):
    example_copy = copy.deepcopy(example)

    example_copy.label = copy.deepcopy(example_copy.predicted_label)

    return example_copy


def compute_mean_logits(list_of_examples, label_vocab):
    """
    Computes the list of mean_logits for each predicted label
    :param list_of_examples: list of examples to consider for the computation
    :param label_vocab: the vocabulary of all possible labels
    :return: the list of mean_logits
    """
    mean_logits_list = []

    for possible_label in label_vocab.itos:
        logits_for_this_label = [example.logit for example in list_of_examples if example.predicted_label ==
                                 possible_label]

        mean_logits_list.append(np.mean(logits_for_this_label, axis=0))

    return mean_logits_list


def add_kl_divergence(list_of_examples, mean_logits, vocab):
    for example in list_of_examples:
        example.kl_divergence = single_kl_computation(example.logit, mean_logits, example.predicted_label, vocab)


def single_kl_computation(logits, mean_logits, predicted_label, vocab):
    corresponding_int = vocab.stoi[predicted_label]

    return entropy(logits, mean_logits[corresponding_int], base=2)
