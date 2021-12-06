from tensorflow import keras
import os
import numpy as np
from PIL import Image, ImageOps
import log

def evaluate_model(model, datapath):
    to_predict = []
    expected = []
    for file in os.listdir(datapath):
        path = os.path.join(datapath, file)
        log.dbg(f'Reading {path}.', 'chomp_simple')

        exp = int(file.split('.')[0][0])
        expected.append(exp)

        img = np.array(ImageOps.invert(Image.open(path).convert('L')))
        to_predict.append(img)

    log.info(f'Shape out custom test data: {np.array(to_predict).shape}')

    prediction = model.predict(keras.utils.normalize(np.array(to_predict)))
    actual = np.argmax(prediction, axis=1)

    accuracy_accum = 0
    confidence_accum = 0
    for i, exp in enumerate(expected):
        accuracy_accum += 1 if exp == actual[i] else 0
        confidence_accum += prediction[i, actual[i]]

    count = len(to_predict)
    log.info(f'Total Accuracy: {round(accuracy_accum / count, 2)}. ' +
             f'Avg Confidence: {round(confidence_accum / count)}\n')
