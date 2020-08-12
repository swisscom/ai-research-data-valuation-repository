"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

import os
import yaml

from dataset.dataset_classes.inference_dataset import InferenceDataset
from dataset.dataset_classes.codi_dataset import CodiDataset
from dataset.dataset_classes.unlabelled_dataset import UnlabelledDataset
from dataset.custom_iterator import CustomIterator

import torch
import spacy

from torchtext import data
from torchtext import datasets


class Dataset:

    def __init__(self, dataset_name, hyper_yaml):
        """
        Creates a DatasetLoader instance.

        :param dataset_name: string which dataset wants to be tested.
            Available datasets: 'mnist', 'cifar10', 'imdb'
        :param hyper_yaml: string name of the yaml file storing the hyperparameters
            The given path should be relative to the path of dataset_loader.py

        Attributes:
            dataset_name: string specifying which dataset to use
            hyper: the dictionary of hyperparameters loaded from a YAML file
            datasets: the datasets used either for training or testing the model
                labelled1: used to train the inference model
                labelled2: used to train the CoDi part
                unlabelled: to be inferred dataset. This is the data we want later to separate between mislabeled or
                    not.
            prediction_logits: the prediction logits from the unlabelled dataset
            data_features: extracted data features that will go again in the CoDi part
            true_indices_codi: the indices of the points of codi_dataset that are correctly labelled during
                the inference part
            size_unlabelled/size_labelled2: size of the unlabelled and labelled2 datasets
            encoder: the encoder used to embed the text data
            model: the neural network model
        """

        script_dir = os.path.dirname(__file__)
        absolute_path = os.path.join(script_dir, hyper_yaml)

        # Load the set of hyperparameters contained in the yaml file
        with open(absolute_path) as f:
            self.hyper = yaml.safe_load(f)

        self.dataset_name = dataset_name

        self.inference_dataset = None
        self.codi_dataset = None
        self.unlabelled_dataset = None

        self.prediction_logits = None
        self.data_features = None

        self.prediction_logits_codi = None
        self.true_indices_codi = None
        self.data_features_codi = None

        self.model = None
        self.text = None
        self.label = None
        self.device = None

    def dataset_to_torch(self):
        """
        Loads the datasets using the torchtext.datasets library.
        There are 3 datasets:
            - the first labelled dataset used to train the inference model
            - the second labelled dataset used to train the classifier
            - the unlabelled dataset for which we flip some portion of the labels, used to test the classification model
        """
        assert self.dataset_name in self.hyper['available_datasets'], 'Dataset given is not valid.'

        # Initial load of the spacy english tokenizer
        spacy.load('en')

        if self.dataset_name == 'imdb':
            TEXT = data.Field(tokenize='spacy', preprocessing=Dataset.generate_bigrams)
            LABEL = data.LabelField(dtype=torch.float)

            print('Loading the IMDB dataset ...')
            train_original, test_original = datasets.IMDB.splits(TEXT, LABEL)

        elif self.dataset_name == 'trec':
            TEXT = data.Field(tokenize='spacy', preprocessing=Dataset.generate_bigrams)
            LABEL = data.LabelField()

            print('Loading the TREC dataset ...')
            train_original, test_original = datasets.TREC.splits(TEXT, LABEL, fine_grained=False)

        if self.hyper['portion_used'] == 1.0:
            train_used = train_original
        else:
            train_used, _ = train_original.split(self.hyper['portion_used'], stratified=True)

        # Beware, the output is train, test, validation
        train1, unlabelled, train2 = train_used.split(split_ratio=self.hyper['train_split_ratio'], stratified=True)

        test1, val1, _ = test_original.split(split_ratio=self.hyper['test_split_ratio'], stratified=True)

        TEXT.build_vocab(train1,
                         max_size=self.hyper['max_vocab_size'],
                         vectors=self.hyper['vocab_vector'],
                         unk_init=torch.Tensor.normal_)

        LABEL.build_vocab(train1)

        self.text = TEXT
        self.label = LABEL

        self.inference_dataset = InferenceDataset(train1, test1, val1)
        self.codi_dataset = CodiDataset(train2)
        self.unlabelled_dataset = UnlabelledDataset(unlabelled)

        print('Size of the inference dataset is ', len(self.inference_dataset.dataset.examples))
        print('Size of the codi dataset is ', len(self.codi_dataset.dataset.examples))
        print('Size of the unlabelled dataset is ', len(self.unlabelled_dataset.dataset.examples))

        print('Available device is :', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

        print('Datasets loaded')

    def update_datasets(self, inference_train, unlabelled_dataset):
        self.inference_dataset.dataset = inference_train
        self.unlabelled_dataset.dataset = unlabelled_dataset

        self.text.build_vocab(self.inference_dataset.dataset, max_size=self.hyper['max_vocab_size'],
                              vectors=self.hyper['vocab_vector'], unk_init=torch.Tensor.normal_)

        self.label.build_vocab(self.inference_dataset.dataset)

    def get_inference_dataset(self):
        return self.inference_dataset.get_dataset()

    def get_inference_pack(self):
        return self.inference_dataset.get_pack()

    def get_codi_dataset(self):
        return self.codi_dataset.get_dataset()

    def get_unlabelled_dataset(self):
        return self.unlabelled_dataset.get_dataset()

    def get_text_label(self):
        return self.text, self.label

    @staticmethod
    def generate_bigrams(x):
        """
        Generate bigrams of words from a words sequence.
        :param x: sequence of words
        :return: sequence of bigrams
        """
        n_grams = set(zip(*[x[i:] for i in range(2)]))
        for n_gram in n_grams:
            x.append(' '.join(n_gram))
        return x

    def get_inference_iterator(self):
        """
        Extracts data for training the inference model in batches
        :return: iterators for the train, validation and test datasets
        """
        train1, test1, val1 = self.get_inference_pack()

        train_iterator, valid_iterator, test_iterator = data.Iterator.splits(
            (train1, val1, test1),
            batch_size=self.hyper['batch_size'],
            sort_within_batch=False,
            shuffle=True,
            sort=False,
            device=self.device)

        return train_iterator, valid_iterator, test_iterator

    def get_unlabelled_iterator(self):
        """
        Extracts unlabelled data and sends it in batches
        :return: iterators for the unlabelled dataset
        """
        unlabelled_iterator = CustomIterator(
            self.unlabelled_dataset.dataset,
            batch_size=self.hyper['batch_size'],
            shuffle=True,
            train=False,
            sort_within_batch=False,
            sort=False,
            device=self.device)

        return unlabelled_iterator

    def get_codi_iterator(self):
        """
        Extracts unlabelled data and sends it in batches
        :return: iterators for the unlabelled dataset
        """
        codi_labelled_iterator = CustomIterator(
            self.codi_dataset.dataset,
            batch_size=self.hyper['batch_size'],
            sort_within_batch=True,
            device=self.device)

        return codi_labelled_iterator

    def get_device(self):
        return self.device


def main():

    dataset_loading = Dataset('imdb', 'yaml_hyper/imdb_hyper.yaml')
    dataset_loading.dataset_to_torch()


if __name__ == '__main__':
    main()
