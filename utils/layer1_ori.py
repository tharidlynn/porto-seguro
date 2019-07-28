import os
from constants import *
from utils import gini, utils, layer2
import gc
import time
import sys

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold

class Layer1Train():
    def __init__(self, model, MODEL_NAME, kinetic_transform=False, drop_stupid=False, cat_transform=False, data_transform=False, recon_category=False,
    feature_interactions=False, engineer_stats=False, remove_outliers=False):
        self.model = model
        self.MODEL_NAME = MODEL_NAME
        self.kinetic_transform = kinetic_transform
        self.drop_stupid = drop_stupid
        self.cat_transform = cat_transform
        self.data_transform = data_transform
        self.recon_category = recon_category
        self.feature_interactions = feature_interactions
        self.engineer_stats = engineer_stats
        self.remove_outliers = remove_outliers

    def make_oof(self, model, X_train, y_train, X_test, MODEL_NAME):
        tmp = time.time()

        assert len(X_train) == len(y_train)
        score = []
        oof_train = np.zeros(X_train.shape[0], )
        oof_test = np.zeros(X_test.shape[0], )
        cat_col = X_train.columns[X_train.columns.str.endswith('cat') == True]
        # Generate Seed numbers for xgb_alphabets only
        # USE SKF_SEED IN THE FUTURES
        # seed = np.random.randint(0, 1000)
        # print('Seed for StratifiedKFold: ', seed)
        skf = StratifiedKFold(n_splits=KFOLDS, random_state=SKF_SEED)
        for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):

            print('\n')

            print('[Fold {}/{} START]'.format(i + 1, KFOLDS))

            X_tr, X_val = X_train.iloc[train_index, :], X_train.iloc[val_index, :]
            y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]


            if self.cat_transform == 'smooth':
                print('Start smoothing encoding')
                for f in cat_col:
                    X_tr[f + "_smooth"], X_val[f + "_smooth"], X_test[f + "_smooth"] = utils.target_encode(
                                                        trn_series=X_tr[f],
                                                        val_series=X_val[f],
                                                        tst_series=X_test[f],
                                                        target=y_tr,
                                                        min_samples_leaf=200,
                                                        smoothing=10,
                                                        noise_level=0
                                                        )

            if 'xgb' in MODEL_NAME:
                print('Xgboost is training')
                eval_set = [(X_tr, y_tr), (X_val, y_val)]
                model.fit(X_tr, y_tr, eval_set=eval_set, eval_metric=gini.gini_xgbsklearn, early_stopping_rounds=100, verbose=True)
                pred_val = model.predict_proba(X_val, ntree_limit=model.best_ntree_limit)[:,1]
                pred_test = model.predict_proba(X_test, ntree_limit=model.best_ntree_limit)[:,1] / KFOLDS

            elif 'lgbm' in MODEL_NAME:
                print('Lightgbm is training')
                eval_set = [(X_tr, y_tr), (X_val, y_val)]
                model.fit(X_tr, y_tr, eval_set=eval_set, eval_metric=gini.gini_lgbsklearn, early_stopping_rounds=70, verbose=True)
                pred_val = model.predict_proba(X_val)[:,1]
                pred_test = model.predict_proba(X_test)[:,1] / KFOLDS

            else:
                print('{} is training'.format(MODEL_NAME))
                if 'rgf' in MODEL_NAME:
                    X_tr = X_tr.fillna(-1)
                    X_val = X_val.fillna(-1)
                    X_test = X_test.fillna(-1)
                    model.fit(X_tr, y_tr)
                elif 'cat' in MODEL_NAME:
                    X_tr = X_tr.replace(np.NaN, -1)
                    X_val = X_val.replace(np.NaN, -1)
                    X_test = X_test.replace(np.NaN, -1)
                    eval_set = [X_val, y_val]
                    model.fit(X_tr, y_tr, eval_set=eval_set, use_best_model=True, verbose=True)
                elif 'fm' in MODEL_NAME:
                    y_tr = y_tr.replace(0, -1)
                    model.fit(X_tr.to_sparse(), y_tr.to_sparse())
                # Use for monitoring overfit in other models
                pred_tr = model.predict_proba(X_tr)[:,1]
                score_tr = gini.eval_gini(y_tr, pred_tr)
                print('[Fold {}/{} Train gini score: {}]'.format(i+1, KFOLDS, score_tr))
                pred_val = model.predict_proba(X_val)[:,1]
                pred_test = model.predict_proba(X_test)[:,1] / KFOLDS

            # For train and validation score to layer 2
            oof_train[val_index] = pred_val
            # Store predictions for submission and oof test
            oof_test += pred_test

            # Store validation score for evaluate model
            score.append(gini.eval_gini(y_val, pred_val))
            gc.collect()
            print('[Fold {}/{} Valid gini score: {}]'.format(i+1, KFOLDS, score[i]))
            print('[Fold {}/{} END]'.format(i+1, KFOLDS))

        print('All score', score)
        print('Average score: {}'.format(np.mean(score)))

        # Export oof_train
        file_path = os.path.join(OOF_PATH, MODEL_NAME + '_train.csv')
        pd.DataFrame({MODEL_NAME: oof_train}).to_csv(file_path, index=False)
        # np.savetxt(file_path, oof_train.reshape(-1, 1), delimiter=',', fmt='%.5f')

        # Export oof_test
        file_path = os.path.join(OOF_PATH, MODEL_NAME + '_test.csv')
        pd.DataFrame({MODEL_NAME: oof_test}).to_csv(file_path, index=False)

        # Make submission files automatically for single leader board
        layer2.make_single_sub(file_path)

        # np.savetxt(file_path, oof_test.reshape(-1, 1), delimiter=',', fmt='%.5f')
        print('SUCCESSFULLY SAVE {} AT {}  PLEASE VERIFY THEM'.format(MODEL_NAME, OOF_PATH))
        print('Training time: {} minutes'.format(str((time.time() - tmp) / 60)))

    def train(self):
        log_path = os.path.join(LOG_PATH, self.MODEL_NAME + '_log.txt')
        orig_stdout = sys.stdout
        f = open(log_path, 'w')
        sys.stdout = f

        combined = utils.load_data()
        if self.kinetic_transform:
            combined = utils.kinetic_transform(combined)
        if self.drop_stupid:
            combined = utils.drop_stupid(combined, type=self.drop_stupid)
        if self.engineer_stats:
            combined = utils.engineer_stats(combined)
        if self.recon_category:
            combined = utils.recon_category(combined)
        if self.cat_transform:
            if self.cat_transform != 'smooth':
                combined = utils.cat_transform(combined, type=self.cat_transform)
        if self.data_transform:
            combined = utils.data_transform(combined,type=self.data_transform)
        if self.feature_interactions:
            combined = utils.feature_interactions(combined)

        train, test = utils.recover_train_test_na(combined, remove_outliers=self.remove_outliers)

        X_train = train.drop('target', axis=1)
        y_train = train.target
        X_test = test

        # Start trainning
        print('Ready to train with:')
        print('Model name ', self.MODEL_NAME)
        print('Model parameters ', self.model.get_params())
        print('X_train shape is', X_train.shape)
        print('y_train shape is', y_train.shape)
        print('X_test shape is', X_test.shape)

        self.make_oof(self.model, X_train, y_train, X_test, self.MODEL_NAME)

        sys.stdout = orig_stdout
        f.close()
