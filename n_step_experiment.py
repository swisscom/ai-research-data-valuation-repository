"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

import numpy as np

from dataset.dataset_loader import Dataset

from inference_models.inference_torch import InferenceTorch
from inference_models.__inference_utils import compute_percent_correct

from codi.nlp_trainer import NLPTrainer

import torch

import time
from inference_models.__inference_utils import epoch_time

"""
This script is designed for the N-step retraining setup.
The inference model is iteratively retrained with more and more points with pseudo-labels.
"""


def main():
    start_time = time.time()

    dataset_name, hyper_yaml = 'trec', 'yaml_hyper/trec_hyper.yaml'
    dataset_loading = Dataset(dataset_name, hyper_yaml)
    dataset_loading.dataset_to_torch()

    print('***IMDB Dataset loaded***')

    inference_train_iterator, inference_val_iterator, inference_test_iterator = dataset_loading.get_inference_iterator()
    text, _ = dataset_loading.get_text_label()

    vocab_size = len(text.vocab)
    pad_idx = text.vocab.stoi[text.pad_token]

    inference_yaml = 'trec.yaml'

    inference_model = InferenceTorch(dataset_name, hyper_yaml, vocab_size, pad_idx, dataset_loading.get_device())
    inference_model.load_from_yaml(inference_yaml)
    initial_acc, initial_f1 = inference_model.train_model(text, inference_train_iterator, inference_val_iterator,
                                                          inference_test_iterator)
    print('***Inference Model trained, now inferring labels.***')

    # This parameter is the number of retraining steps
    number_steps = 10

    # Those arrays store the scores for performance metrics
    initial_accuracies = np.array([initial_acc, initial_f1])
    retraining_accuracies = np.zeros(number_steps)
    percent_correct_array = np.zeros(number_steps)
    size_indices_array = np.zeros(number_steps)

    retraining_f1 = np.zeros(number_steps)

    for step in range(number_steps):
        print('Starting step {}'.format(step))

        unlabelled_iterator = dataset_loading.get_unlabelled_iterator()
        inference_model.infer_labels(unlabelled_iterator)
        print('Percent correct on unlabelled dataset prediction ',
              compute_percent_correct(dataset_loading.get_unlabelled_dataset().examples))

        print('***Unlabelled dataset processed, now processing CoDi dataset.***')

        codi_labelled_iterator = dataset_loading.get_codi_iterator()
        inference_model.infer_labels(codi_labelled_iterator)

        print('***CoDi labelled dataset processed.***')

        codi_trainer = NLPTrainer(yaml_model_path='codi/mlp_codi.yaml', yaml_train_path='codi/nlp_trainer_n_step.yaml')

        codi_trainer.create_labelled_dataset(codi_labelled_iterator)
        codi_trainer.train()
        print('***CoDi model trained.***')

        _ = codi_trainer.create_unlabelled_dataset(unlabelled_iterator)
        print('***Prediction done.***')

        # Beginning of retraining experiments
        unlabelled_dataset_original = dataset_loading.get_unlabelled_dataset()
        inference_train_original = dataset_loading.get_inference_dataset()

        start_step = time.time()

        # The exp1_percentage is the percentage of points taken at each step
        # In this case, it is a fixed value, but could be changed step after step
        inference_train, _, unlabelled_dataset, percent_correct, size_indices = \
            inference_model.process_retraining(inference_train_original, unlabelled_dataset_original, 0,
                                               exp1_percentage=0.1)

        percent_correct_array[step] = percent_correct
        size_indices_array[step] = size_indices
        print('Percent Correct for this threshold is {}, for {} points'.format(percent_correct, size_indices))

        dataset_loading.update_datasets(inference_train, unlabelled_dataset)

        inference_train_iterator, inference_val_iterator, inference_test_iterator = dataset_loading. \
            get_inference_iterator()
        text, _ = dataset_loading.get_text_label()
        vocab_size = len(text.vocab)

        del inference_model
        torch.cuda.empty_cache()
        inference_model = InferenceTorch(dataset_name, hyper_yaml, vocab_size, pad_idx, dataset_loading.get_device())
        inference_model.load_from_yaml(inference_yaml)

        test_acc, test_f1 = inference_model.train_model(text, inference_train_iterator, inference_val_iterator,
                                                        inference_test_iterator)

        retraining_accuracies[step] = test_acc
        retraining_f1[step] = test_f1

        end_step = time.time()

        step_mins, step_secs = epoch_time(start_step, end_step)
        print(f'Time for step {step}: {step_mins}m {step_secs}s')

        torch.cuda.empty_cache()

        np.save('results/initial_accuracies', initial_accuracies)
        np.save('results/retraining_experiment', retraining_accuracies)
        np.save('results/retraining_f1', retraining_f1)

        np.save('results/percent_correct', percent_correct_array)
        np.save('results/size_indices', size_indices_array)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Total time: {epoch_mins}m {epoch_secs}s')


if __name__ == '__main__':
    main()
