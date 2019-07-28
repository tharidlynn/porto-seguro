from utils import utils, layer1
import time

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

MODEL_NAME = 'knn_2048_40_feat'

def main():
    train, test = utils.load_data()
    train, test = utils.clean_data(train, test)
    train, test = utils.engineer_features(train, test)

    # Prepare X and y following with parsing to numpy datatype
    top_importance = ['ps_reg_03',
                     'ps_car_13',
                     'ps_ind_03',
                     'ps_car_14',
                     'ps_ind_15',
                     'ps_reg_02',
                     'ps_ind_01',
                     'ps_reg_01',
                     'ps_car_15',
                     'ps_car_12',
                     'ps_ind_05_cat_0.0',
                     'ps_ind_17_bin',
                     'ps_ind_07_bin',
                     'ps_ind_16_bin',
                     'ps_car_11',
                     'ps_car_09_cat_1.0',
                     'ps_ind_06_bin',
                     'ps_car_07_cat_0.0',
                     'ps_ind_04_cat_0.0',
                     'ps_ind_02_cat_1.0',
                     'ps_car_01_cat_7.0',
                     'ps_ind_05_cat_6.0',
                     'ps_car_09_cat_0.0',
                     'ps_car_01_cat_9.0',
                     'ps_ind_05_cat_2.0',
                     'ps_ind_09_bin',
                     'ps_car_05_cat_0.0',
                     'ps_ind_08_bin',
                     'ps_car_01_cat_11.0',
                     'ps_ind_02_cat_2.0',
                     'ps_car_04_cat_2',
                     'ps_car_04_cat_0',
                     'ps_car_09_cat_2.0',
                     'ps_car_08_cat_0',
                     'ps_car_02_cat_0.0',
                     'ps_car_01_cat_4.0',
                     'ps_car_07_cat_1.0',
                     'ps_car_06_cat_1',
                     'ps_car_06_cat_9',
                     'ps_ind_04_cat_1.0']

    X_train = train[top_importance].values
    y_train = train.target.values
    X_test = test[top_importance].values

    # Data standarization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Data
    params = {
        'n_neighbors': 2048,
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
