"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

from subprocess import call
import numpy as np
from codi.codi_utils import create_speech_data, create_unlabelled_speech_data, save_ids
from codi.speech_trainer import SpeechTrainer


def train_codi(labelling='naive', threshold=None):
    """
    Process of training : computing features, creating data, training.
    Params: labelling : 'naive ' or 'levenshtein'
            threshold : for the levenshtein method.
    """
    codi_trainer = SpeechTrainer(yaml_model_path='codi/mlp_codi.yaml',
                                 yaml_train_path='codi/speech_trainer.yaml')

    X, y = create_speech_data(codi_trainer.train_params['features'], method=labelling, thresh=threshold)
    y = y[~np.isnan(X).any(axis=1)]
    X = X[~np.isnan(X).any(axis=1)]

    codi_trainer.create_labelled_dataset(X, y)
    codi_trainer.train()

    print('***CoDi model trained.***')


def infer_from_codi():
    """
    Process of inferring : computing features, creating data, inferring, filtering points.
    """
    codi_trainer = SpeechTrainer(yaml_model_path='codi/mlp_codi.yaml',
                                 yaml_train_path='codi/speech_trainer.yaml')

    X = create_unlabelled_speech_data(codi_trainer.train_params['features'])
    X = X[~np.isnan(X).any(axis=1)]

    codi_trainer.create_unlabelled_dataset(X)
    save_ids(codi_trainer.test())
    print('***Prediction done.***')


def experiment():
    """
    Computes the experiment based on the value in train_codi.yaml
    exp : '1' is the fixed-sized variable threshold experiment.
    exp : '2' is the variable-sized variable threshold experiment.
    """
    ids_trust = np.load('codi/outputs_init/ids_trust.npy', allow_pickle=True)
    ids_no_trust = np.load('codi/outputs_init/ids_no_trust.npy', allow_pickle=True)
    for i in range(0, len(ids_trust)):
        np.save('codi/outputs/ids_trust.npy', ids_trust[i])
        np.save('codi/outputs/ids_no_trust.npy', ids_no_trust[i])
        call('cd s5 && ./run.sh --stage 23 --i {}'.format(i+1), shell=True)


def iterative_process(N):
    """
    Iterative retraining (N iterations)
    """
    for i in range(0, N):
        call('cd speech_inference && ./run.sh --stage 23 --i {}'.format(i+1), shell=True)
        infer_from_codi()


def main():
    """
    Whole Process pipeline encompassing the other modules.
    """
    # If GMM has not already been trained on labelled set, put --stage 0,
    # otherwise, put --stage 20
    call('cd speech_inference && ./run.sh --stage 0', shell=True)
    print('***Inference model trained***')

    train_codi(labelling='levenshtein', threshold=None)
    infer_from_codi()

    # experiment()

    iterative_process(N=10)


if __name__ == '__main__':
    main()
