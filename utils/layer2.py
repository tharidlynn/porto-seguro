import glob
import os
from constants import *

import pandas as pd

def get_oof(mode):
    # m = observations n = numbers of models
    if mode == 'train':
        print('Get train mode')
        # List of csv path
        csv_list = glob.glob(OOF_PATH + '/*_train.csv')
    elif mode == 'test':
        print('Get test mode')
        # List of csv path
        csv_list = glob.glob(OOF_PATH + '/*_test.csv')
    # Base meta features dataframe
    meta_features_df = pd.read_csv(csv_list[0])
    for f in csv_list[1:]:
        tmp_df = pd.read_csv(f)
        meta_features_df = pd.concat([meta_features_df, tmp_df], axis=1)

    return meta_features_df


def make_single_sub(path):
    print(path)
    test = pd.read_csv(path)
    test_id = pd.read_csv(DATA_TEST_PATH).id

    model_name = test.columns.values[0]
    test.columns.values[0] = 'target'

    sub = pd.concat([test_id, test], axis=1)
    print(sub.head(20))
    file_path = os.path.join(SUBMISSION_PATH, model_name + '_sub.csv')
    sub.to_csv(file_path, index=False)

    print('Successfully make {} submission file'.format(model_name))
