#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import numpy as np
from evaluation import evaluate_model
from typing import Union
import sys

from cacher import cached
import log


@cached
def get_deep_trained_model(norm_xtrain: Union[np.ndarray, tf.Tensor],
                           ytrain: Union[np.ndarray, tf.Tensor]) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(norm_xtrain.shape[1],
                                          norm_xtrain.shape[2])),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(10, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(norm_xtrain, ytrain, epochs=5)

    keras.utils.plot_model(model, to_file=f'output/chomp_deep.model.png')

    return model, history


@cached
def get_dft(xtrain):
    return tf.signal.fft2d(xtrain)


if __name__ == '__main__':
    # Loading the data.
    log.info('(Down)loading Data.')
    (xtrain, ytrain), (xtest, ytest) = keras.datasets.mnist.load_data()
    log.info(f'Training data size: {xtrain.shape},{ytrain.shape}')
    log.info(f'Test data size: {xtest.shape},{ytest.shape}')

    norm_xtrain = keras.utils.normalize(xtrain)
    norm_xtest = keras.utils.normalize(xtest)

    model, history = get_deep_trained_model(norm_xtrain, ytrain)

    # Test
    log.info('Testing with the MNIST test data:')
    loss, accuracy = model.evaluate(norm_xtest, ytest)
    log.info(f'Loss: {round(loss, 2)}, Accuracy: {round(accuracy, 2)}.')

    # Let us now attempt the DFT training.
    log.info('Calculating the DFTs')
    xtrain_dft = get_dft(xtrain)
    xtest_dft = get_dft(xtest)
    log.info(f'The shape of the DFT is {xtrain_dft.shape}, ' +
             f'testing: {xtest_dft.shape}')

    # Reshape and normalize
    dims = (tf.shape(xtrain_dft)[0], tf.shape(xtrain)[1]**2)
    norm_xtrain_dft = keras.utils.normalize(xtrain_dft)
    norm_xtest_dft = keras.utils.normalize(xtest_dft)

    dft_model, dft_history = get_deep_trained_model(norm_xtrain_dft, ytrain)

    log.info('Testing DFT with the MNIST test data:')
    loss, accuracy = dft_model.evaluate(norm_xtest_dft, ytest)
    log.info(f'Loss: {round(loss, 2)}, Accuracy: {round(accuracy, 2)}.')

    # After everything is calculated, let's try our shot at a custom dataset.
    evaluate_model(model, sys.argv[1])
    evaluate_model(dft_model, sys.argv[1])
