from constants import SEED
from utils import utils, layer1

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier

MODEL_NAME = 'extratree'

def main():
    train, test = utils.load_data()
    train, test = utils.engineer_stats(train, test)
    train, test = utils.engineer_features(train, test)

    X_train = train.drop(['id', 'target'], axis=1).values
    y_train = train.target.values
    X_test = test.drop('id', axis=1).values

    # Train Data
    params = {
        'max_depth': 10,
        'max_features': 'sqrt',
        'min_samples_split': 100,
        'min_samples_leaf': 50,
        'n_estimators': 800,
        'n_jobs': -1,
        'oob_score': False,
        'random_state': SEED
    }
    model = ExtraTreesClassifier(**params)

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
