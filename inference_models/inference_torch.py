"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

import os
import yaml
import numpy as np

from inference_models.inference_model import InferenceModel
from inference_models.fast_text import FastText

import torch
from torch import nn
from torch import optim
import torchtext

import inference_models.__inference_utils as utils
import time
import random
import copy


class InferenceTorch(InferenceModel):

    def __init__(self, dataset_name, hyper_yaml, vocab_size, pad_idx, device):
        """
        Inits an InferenceTorch model.
        Also loads the hyperparameters from a YAML file.
        :param dataset_name: name of the dataset we want to work on (for now, only 'imdb')
        :param hyper_yaml: path to the YAML file where the hyperparameters are stored
        :param vocab_size: vocabulary size used
        :param pad_idx: padding index we want to use for padding
        :param device: the device used to compute (either CPU or CUDA if we work with GPU or not).
        """
        super().__init__()

        self.dataset_name = dataset_name
        self.hyper_yaml = hyper_yaml

        script_dir = os.path.dirname(__file__)
        absolute_path = os.path.join(script_dir, hyper_yaml)

        # Load the set of hyperparameters contained in the yaml file
        with open(absolute_path) as f:
            self.hyper = yaml.safe_load(f)

        self.binary = self.hyper['binary']

        self.model = None
        self.optimizer = None
        self.criterion = None

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.device = device

    def load_from_yaml(self, yaml_file):
        """
        Loads the model from a YAML file.
        The global structure is defined in this method, and all the parameters are listed in the YAML file, to be
        easily accessible.
        :param yaml_file: path to the YAML file where we store the
        """
        # load hyperparameters of the NN model into param --> need a FullLoader here to load tuples
        script_dir = os.path.dirname(__file__)
        absolute_path = os.path.join(script_dir, yaml_file)

        with open(absolute_path, 'r') as f:
            param = yaml.load(f, Loader=yaml.FullLoader)

        self.model = FastText(self.vocab_size,
                              param['embedding_layer']['embedding_dim'],
                              param['final_layer']['output_dim'],
                              self.pad_idx)

        self.optimizer = optim.Adam(self.model.parameters())

    def train_model(self, text, train_iterator, valid_iterator, test_iterator):
        """
        Trains the inference model using the Labelled dataset 1.
        Also stores the model state after training in the save checkpoint given in the hyperparameter YAML file.
        :param text:
        :param train_iterator: train dataset in a batch iterator form
        :param valid_iterator: validation dataset in a batch iterator form
        :param test_iterator: test dataset in a batch iterator form
        """
        pretrained_embeddings = text.vocab.vectors

        self.model.embedding.weight.data.copy_(pretrained_embeddings)

        unknown_index = text.vocab.stoi[text.unk_token]

        self.model.embedding.weight.data[unknown_index] = torch.zeros(self.hyper['embedding_dim'])
        self.model.embedding.weight.data[self.pad_idx] = torch.zeros(self.hyper['embedding_dim'])

        self.criterion = getattr(nn, self.hyper['criterion'])()

        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        best_valid_acc = 0

        for epoch in range(self.hyper['epochs']):
            start_time = time.time()

            train_loss, train_acc = utils.train(self.model, train_iterator, self.optimizer, self.criterion,
                                                binary=self.binary)
            valid_loss, valid_acc, _ = utils.evaluate(self.model, valid_iterator, self.criterion, binary=self.binary)

            end_time = time.time()

            epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(self.model.state_dict(), self.hyper['model_saved'])

            if epoch % 10 == 0:
                print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

        self.model.load_state_dict(torch.load(self.hyper['model_saved']))

        test_loss, test_acc, test_f1 = utils.evaluate(self.model, test_iterator, self.criterion, binary=self.binary)
        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)
        print('Test f1-score:', test_f1)

        return test_acc, test_f1

    def infer_labels(self, iterator):
        """
        Infer prediction logits for the given dataset.
        All modifications are done
        :param iterator: the given dataset for which we want to predict labels in batch iterator form.
        """
        utils.predict(self.model, iterator, binary=self.binary)

        # Add the KL Divergence feature
        mean_logits = utils.compute_mean_logits(iterator.dataset.examples, iterator.dataset.fields['label'].vocab)
        utils.add_kl_divergence(iterator.dataset.examples, mean_logits, iterator.dataset.fields['label'].vocab)

    @staticmethod
    def process_retraining(inference_dataset, unlabelled_dataset, threshold_index, exp1_size=None,
                           exp1_percentage=None, adaptation=False, take_random=False, random_percentage=None):
        """
        Wraps up the retraining process.
        That means adding inferred points to the inference dataset and removing them from the unlabelled dataset.
        :param inference_dataset: inference dataset before adding points of the retraining
        :param unlabelled_dataset: unlabelled dataset
        :param threshold_index: index for the threshold we want to use
        :param exp1_size: number of points we want to keep if in experiment 1 setup (that is random sample amongst
        all selected points).
            If nothing is given for this parameter, then all points are taken.
        :param exp1_percentage:
        :param adaptation: whether we want to generate the inferences dataset with :
            - the original inference dataset + new points if 'adaptation' is set to False
            - only the new points if 'adaptation' is set to True
            in both cases, the selected points will be removed from the unlabelled dataset
        :param take_random: whether we want to select points randomly for retraining
        :param random_percentage: percentage of points to take if take_random is set to True

        :return:
            - new_inference_dataset: the inference dataset with new points added
            - new_inference_god_dataset: the inference dataset with points added with their true labels
            - new_unlabelled_dataset: the unlabelled dataset from which selected points have been removed
            - percent_correct: the percentage of points correct in those added to the inference dataset
            - size_indices: the sizes of the indices sets codi gave for each threshold
        """
        new_inference_dataset = torchtext.data.Dataset(copy.deepcopy(inference_dataset.examples),
                                                       inference_dataset.fields)
        new_inference_god_dataset = torchtext.data.Dataset(copy.deepcopy(inference_dataset.examples),
                                                           inference_dataset.fields)
        new_unlabelled_dataset = torchtext.data.Dataset(copy.deepcopy(unlabelled_dataset.examples),
                                                        unlabelled_dataset.fields)

        if take_random:
            all_indices = list(np.arange(len(new_unlabelled_dataset.examples)))
            size_to_take = round(random_percentage * len(all_indices))
            corresponding_indices = random.sample(all_indices, size_to_take)

        else:
            corresponding_indices = [index for index, example in enumerate(new_unlabelled_dataset.examples) if
                                     example.is_accepted[threshold_index]]

            if exp1_size is not None:
                corresponding_indices = random.sample(corresponding_indices, exp1_size)
            elif exp1_percentage is not None:
                size_to_take = round(exp1_percentage * len(corresponding_indices))
                corresponding_indices = random.sample(corresponding_indices, size_to_take)

        size_indices = len(corresponding_indices)

        # Remove from unlabelled dataset
        selected_unlabelled_points = np.array(new_unlabelled_dataset.examples)[corresponding_indices]

        new_examples = [example for ind, example in enumerate(new_unlabelled_dataset.examples) if ind not in
                        corresponding_indices]
        new_unlabelled_dataset.examples = new_examples

        # Compute percent correct
        percent_correct = utils.compute_percent_correct(selected_unlabelled_points)

        # Add to inference dataset
        selected_points_list = selected_unlabelled_points.tolist()

        if adaptation:
            new_inference_god_dataset.examples = selected_points_list
            new_inference_dataset.examples = [utils.prediction_example(example) for example in
                                              selected_points_list.copy()]

        else:
            new_inference_god_dataset.examples = new_inference_god_dataset.examples + selected_points_list

            new_inference_dataset.examples = new_inference_dataset.examples + [utils.prediction_example(example) for
                                                                               example in selected_points_list.copy()]

        return new_inference_dataset, new_inference_god_dataset, new_unlabelled_dataset, percent_correct, size_indices
