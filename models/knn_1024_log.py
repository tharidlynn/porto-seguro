from utils import utils, layer1
import time

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

MODEL_NAME = 'knn_1024_log'
SEED = 22

def main():
    train, test = utils.load_data()
    train, test = utils.clean_data(train, test)
    train, test = utils.engineer_features(train, test)

    # Apply Log transformation log(x+1)
    train[train.drop(['id', 'target'], axis=1).columns] = train.drop(['id', 'target'], axis=1).apply(np.log1p)
    test[test.drop('id', axis=1).columns] = test.drop('id', axis=1).apply(np.log1p)

    # Prepare X and y following with parsing to numpy datatype
    X_train = train.drop(['id', 'target'], axis=1).values
    y_train = train.target.values
    X_test = test.drop('id', axis=1).values

    # Train Data
    params = {
        'n_neighbors': 1024,
        'n_jobs': -1
        }
    model = KNeighborsClassifier(**params)

    # Start trainning
    print('Ready to train with:')
    print('Model name ', MODEL_NAME)
    print('Model parameters ', model)
    print('X_train shape is', X_train.shape)
    print('y_train shape is', y_train.shape)
    print('X_test shape is', X_test.shape)
    tmp = time.time()
    layer1.make_oof(model, X_train, y_train, X_test, MODEL_NAME)
    print('Total run time: {} seconds'.format(str(time.time() - tmp)))

if __name__ == '__main__':
    main()
