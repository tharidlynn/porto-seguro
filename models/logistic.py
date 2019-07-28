from utils import utils, layer1

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

MODEL_NAME = 'logistic'

def main():
    combined = utils.load_data()
    combined = utils.drop_stupid(combined)
    combined = utils.cat_transform(combined, type='onehot')

    combined.replace(-1, combined.median(axis=0), inplace=True)
    combined = utils.data_transform(combined, 'log')

    train, test = utils.recover_train_test_na(combined, fillna=False)

    # Prepare X and y following with parsing to numpy datatype
    X_train = train.drop('target', axis=1).values
    y_train = train.target.values
    X_test = test.values

    # Train Data
    params = {
        'class_weight': 'balanced',
        'verbose': 1
    }
    model = LogisticRegression(**params)

    # Start trainning
    print('Ready to train with:')
    print('Model name ', MODEL_NAME)
    print('Model parameters ', model)
    print('X_train shape is', X_train.shape)
    print('y_train shape is', y_train.shape)
    print('X_test shape is', X_test.shape)
    layer1.make_oof(model, X_train, y_train, X_test, MODEL_NAME)

if __name__ == '__main__':
    main()
