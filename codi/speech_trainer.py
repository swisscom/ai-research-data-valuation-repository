"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

from pickle import dump, load
import tensorflow as tf
import numpy as np

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

from codi.mlp_codi import MLPCoDi
from codi.codi_utils import generate_plots
from codi.codi_utils import filter_datapoints

from codi.codi_trainer import CoDiTrainer


class SpeechTrainer(MLPCoDi, CoDiTrainer):
    """
    Creates a CoDiTrainer instance.

    Parameters:
        name: string
        yaml_model_path: path of yaml containing model params.
        yaml_train_path: path to yaml containing training params.

    Attributes:
        datasets: train_dataset, test_dataset, validation_dataset, unlabelled_dataset
        scaler: the scaler instance ued to scale train, val and test sets.
        train_params: parameters extracted from yaml
        class_weights: class weights computed for imbalanced classes.
    """
    def __init__(self, name='name', yaml_model_path='mlp_codi.yaml',
                 yaml_train_path='train_codi.yaml'):
        super(SpeechTrainer, self).__init__(name=name, yaml_path=yaml_model_path)
        self.load_model()
        self.train_params = self.load_yaml(yaml_train_path)
        self.scaler = None
        self.class_weights = None
        self.train_dataset = None
        self.test_dataset = None
        self.validation_dataset = None
        self.unlabelled_dataset = None
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

    def train(self):
        """
        Trains the model.
        """
        es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=self.train_params['patience'],
                                           mode='auto',
                                           baseline=None, restore_best_weights=True)

        loss_fn = keras.losses.MeanSquaredError(reduction='sum_over_batch_size')
        self.compile(loss=loss_fn, optimizer=self.train_params['optimizer'],
                     metrics=[keras.metrics.MeanSquaredError()])

        history = self.fit(self.train_dataset, validation_data=self.validation_dataset,
                           epochs=self.train_params['epochs'],
                           class_weight=self.class_weights, callbacks=es)
        test_score = self.evaluate(self.test_dataset)

        self.save_weights(self.train_params['path_to_ckpt'])

        generate_plots([history.history['mean_squared_error'], history.history['loss'],
                        history.history['val_loss'], history.history['val_mean_squared_error']])

        return history.history['mean_squared_error'][-1], test_score[-1]

    def create_labelled_dataset(self, X, y):
        """
        Creates labelled dataset as tf.data.dataset object.

        Args:
            X: data, output of feature extractor block
            y: target (levenshtein distance between decoded and true text)
        """
        y = y.astype(np.float32)

        self.scaler = StandardScaler()

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=0.1,
                                                                              random_state=42)
        self.class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                                              np.unique(self.y_train),
                                                                              self.y_train)))
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_val = self.scaler.transform(self.x_val)
        self.x_test = self.scaler.transform(self.x_test)

        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        self.train_dataset = self.train_dataset.shuffle(buffer_size=512).batch(self.train_params['batch_size'])

        self.validation_dataset = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val))
        self.validation_dataset = self.validation_dataset.batch(self.train_params['test_batch_size'])

        self.test_dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
        self.test_dataset = self.test_dataset.batch(self.train_params['test_batch_size'])

        dump(self.scaler, open(self.train_params['path_to_scaler'], 'wb'))

    def create_unlabelled_dataset(self, X):
        """
        Creates unlabelled dataset as tf.data.dataset object.
        """
        self.scaler = load(open(self.train_params['path_to_scaler'], 'rb'))
        X = self.scaler.transform(X)
        self.unlabelled_dataset = tf.data.Dataset.from_tensor_slices(X)
        self.unlabelled_dataset = self.unlabelled_dataset.batch(self.train_params['test_batch_size'])

    def test(self):
        """
        Classifies (evaluates) new unlabelled data using the previously trained model.

        Args:
            X: unlabelled data, output of feature extractor block
        """
        # Load the state of the old model
        self.load_weights(self.train_params['path_to_ckpt']).expect_partial()
        test_preds = self.predict(self.unlabelled_dataset)
        return filter_datapoints(test_preds, self.train_params['filtering'])
