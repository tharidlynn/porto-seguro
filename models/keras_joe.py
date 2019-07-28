
# coding: utf-8

# https://www.kaggle.com/tilii7/keras-averaging-runs-gini-early-stopping


from utils import utils, gini
import time
from constants import *
import os
import gc

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold

from keras.models import load_model, Sequential
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Reshape, Concatenate, Merge
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, CSVLogger
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
from keras.layers.embeddings import Embedding

from tensorflow import set_random_seed
set_random_seed(88)



np.random.seed(88) # for reproducibility
MODEL_NAME = 'keras_joe'
SEED = 88

combined = utils.load_data()
# combined = utils.bojan_engineer(combined)
# combined = utils.drop_stupid(combined)
# combined = utils.engineer_stats(combined)
# combined = utils.recon_category(combined)
# combined = utils.cat_transform(combined, 'onehot')
# combined = utils.data_transform(combined, self.data_transform)
# combined = utils.feature_interactions(combined)
train, test = utils.recover_train_test_na(combined, fillna=False)


# Fillna for minmax scaler
train = train.replace(np.NaN, -1)
test = test.replace(np.NaN, -1)

X_train = train.drop('target', axis=1)
y_train = train.target
X_test = test

cols_use = [c for c in X_train.columns if (not c.startswith('ps_calc_'))]

X_train = X_train[cols_use]
X_test = X_test[cols_use]

col_vals_dict = {c: list(X_train[c].unique()) for c in X_train.columns if c.endswith('_cat')}

embed_cols = []
for c in col_vals_dict:
    if len(col_vals_dict[c])>2:
        embed_cols.append(c)
        print(c + ': %d values' % len(col_vals_dict[c])) #look at value counts to know the embedding dimensions

print('\n')

class gini_callback(Callback):
    def __init__(self, training_data, validation_data):
        self.X_tr = training_data[0]
        self.y_tr = training_data[1]
        self.X_val = validation_data[0]
        self.y_val = validation_data[1]
        self.best_lap = 0

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_tr = self.model.predict_proba(self.X_tr)
        logs['gini_tr'] = gini.gini_sklearn(self.y_tr, y_pred_tr)
        y_pred_val = self.model.predict_proba(self.X_val)
        logs['gini_val'] = gini.gini_sklearn(self.y_val, y_pred_val)

        # if logs['gini_val'] > self.best_lap:
        #     self.best_lap = logs['gini_val']

        #     global pred_val, pred_test
        #     pred_val = y_pred_val
        #     pred_test = self.model.predict_proba(X_test)

        print('Gini Score in training set: {},  test set: {}'.format(logs['gini_tr'], logs['gini_val']))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def preproc(X_train, X_val, X_test):

    input_list_train = []
    input_list_val = []
    input_list_test = []

    #the cols to be embedded: rescaling to range [0, # values)
    for c in embed_cols:
        raw_vals = np.unique(X_train[c])
        val_map = {}
        for i in range(len(raw_vals)):
            val_map[raw_vals[i]] = i
        input_list_train.append(X_train[c].map(val_map).values)
        input_list_val.append(X_val[c].map(val_map).fillna(0).values)
        input_list_test.append(X_test[c].map(val_map).fillna(0).values)

    #the rest of the columns
    other_cols = [c for c in X_train.columns if (not c in embed_cols)]
    input_list_train.append(X_train[other_cols].values)
    input_list_val.append(X_val[other_cols].values)
    input_list_test.append(X_test[other_cols].values)

    return input_list_train, input_list_val, input_list_test


def create_model():
    models = []

    model_ps_ind_02_cat = Sequential()
    model_ps_ind_02_cat.add(Embedding(5, 3, input_length=1))
    model_ps_ind_02_cat.add(Reshape(target_shape=(3,)))
    models.append(model_ps_ind_02_cat)

    model_ps_ind_04_cat = Sequential()
    model_ps_ind_04_cat.add(Embedding(3, 2, input_length=1))
    model_ps_ind_04_cat.add(Reshape(target_shape=(2,)))
    models.append(model_ps_ind_04_cat)

    model_ps_ind_05_cat = Sequential()
    model_ps_ind_05_cat.add(Embedding(8, 5, input_length=1))
    model_ps_ind_05_cat.add(Reshape(target_shape=(5,)))
    models.append(model_ps_ind_05_cat)

    model_ps_car_01_cat = Sequential()
    model_ps_car_01_cat.add(Embedding(13, 7, input_length=1))
    model_ps_car_01_cat.add(Reshape(target_shape=(7,)))
    models.append(model_ps_car_01_cat)

    model_ps_car_02_cat = Sequential()
    model_ps_car_02_cat.add(Embedding(3, 2, input_length=1))
    model_ps_car_02_cat.add(Reshape(target_shape=(2,)))
    models.append(model_ps_car_02_cat)

    model_ps_car_03_cat = Sequential()
    model_ps_car_03_cat.add(Embedding(3, 2, input_length=1))
    model_ps_car_03_cat.add(Reshape(target_shape=(2,)))
    models.append(model_ps_car_03_cat)

    model_ps_car_04_cat = Sequential()
    model_ps_car_04_cat.add(Embedding(10, 5, input_length=1))
    model_ps_car_04_cat.add(Reshape(target_shape=(5,)))
    models.append(model_ps_car_04_cat)

    model_ps_car_05_cat = Sequential()
    model_ps_car_05_cat.add(Embedding(3, 2, input_length=1))
    model_ps_car_05_cat.add(Reshape(target_shape=(2,)))
    models.append(model_ps_car_05_cat)

    model_ps_car_06_cat = Sequential()
    model_ps_car_06_cat.add(Embedding(18, 8, input_length=1))
    model_ps_car_06_cat.add(Reshape(target_shape=(8,)))
    models.append(model_ps_car_06_cat)

    model_ps_car_07_cat = Sequential()
    model_ps_car_07_cat.add(Embedding(3, 2, input_length=1))
    model_ps_car_07_cat.add(Reshape(target_shape=(2,)))
    models.append(model_ps_car_07_cat)

    model_ps_car_09_cat = Sequential()
    model_ps_car_09_cat.add(Embedding(6, 3, input_length=1))
    model_ps_car_09_cat.add(Reshape(target_shape=(3,)))
    models.append(model_ps_car_09_cat)

    model_ps_car_10_cat = Sequential()
    model_ps_car_10_cat.add(Embedding(3, 2, input_length=1))
    model_ps_car_10_cat.add(Reshape(target_shape=(2,)))
    models.append(model_ps_car_10_cat)

    model_ps_car_11_cat = Sequential()
    model_ps_car_11_cat.add(Embedding(104, 10, input_length=1))
    model_ps_car_11_cat.add(Reshape(target_shape=(10,)))
    models.append(model_ps_car_11_cat)

    model_rest = Sequential()
    model_rest.add(Dense(16, input_dim=24))
    models.append(model_rest)

    model = Sequential()
    model.add(Merge(models, mode='concat'))
    model.add(Dense(80))
    model.add(Activation('relu'))
    model.add(Dropout(.35))
    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(.15))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model




epochs = 1024
batch_size = 4096
patience = 10
KFOLDS = 5
runs_per_fold =3


tmp = time.time()
skf = StratifiedKFold(n_splits=KFOLDS, random_state=SEED)
scores = []
oof_train = np.zeros((X_train.shape[0],1))
oof_test = np.zeros((X_test.shape[0],1))


for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
    assert len(X_train) == len(y_train)

    score_fold = []

    print('\n')

    print('[Fold {}/{} START]'.format(i + 1, KFOLDS))

    X_tr, X_val = X_train.iloc[train_index,:], X_train.iloc[val_index,:]
    y_tr, y_val = y_train[train_index], y_train[val_index]

    #upsampling adapted from kernel:
    #https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283
    pos = (pd.Series(y_tr == 1))

    # Add positive examples
    X_tr = pd.concat([X_tr, X_tr.loc[pos]], axis=0)
    y_tr = pd.concat([y_tr, y_tr.loc[pos]], axis=0)

    # Shuffle data
    idx = np.arange(len(X_tr))
    np.random.shuffle(idx)
    X_tr = X_tr.iloc[idx]
    y_tr = y_tr.iloc[idx]

    #preprocessing
    X_tr, X_val, X_test = preproc(X_tr, X_val, test)

    for j in range(runs_per_fold):
        print('Starting run {}'.format(j+1))

        pred_val = np.zeros((len(val_index),1))
        pred_test = np.zeros((892816,1))
        log_path = os.path.join(LOG_PATH, MODEL_NAME + '_log.csv')
        checkpoint_path = os.path.join(LOG_PATH, MODEL_NAME + '_check.check'.format(j))

        callbacks = [
        gini_callback(training_data=(X_tr, y_tr), validation_data=(X_val, y_val)),
        EarlyStopping(monitor='gini_val', patience=patience, mode='max', verbose=1),
        CSVLogger(log_path, separator=',', append=False),
        ModelCheckpoint(checkpoint_path, monitor='gini_val', mode='max', save_best_only=True, save_weights_only=True, verbose=1)
        ]

        model = create_model()


        model.fit(X_tr, y_tr, shuffle=False, batch_size=batch_size, epochs=epochs, verbose=99, callbacks=callbacks)

        # delete current model
        del model

        # load best model of each run
        model = create_model()
        model.load_weights(checkpoint_path, by_name=False)

        # For train and valid only
        pred_val = model.predict_proba(X_val)
        oof_train[val_index] += pred_val / runs_per_fold

        # Store average score for evaluate model
        score_fold.append(gini.gini_sklearn(y_val, pred_val))

        print('Run {}: {}'.format(j+1, score_fold[j]))

        pred_test_lap = model.predict_proba(X_test)
        pred_test += pred_test_lap / runs_per_fold

    # Store test predictions for submissions

    oof_test += pred_test / KFOLDS

    scores.append(np.mean(score_fold))
    print('[Fold {}/{} Gini score: {}]'.format(i+1, KFOLDS, scores[i]))

    gc.collect()
    print('[Fold {}/{} END]'.format(i+1, KFOLDS))

print('Average score: {}'.format(np.mean(scores)))
print('Total run time: {} seconds'.format(time.time() - tmp))

# Export oof_train
file_path = os.path.join(OOF_PATH, MODEL_NAME + '_train.csv')
pd.DataFrame({MODEL_NAME: oof_train.reshape(-1, )}).to_csv(file_path, index=False)
# np.savetxt(file_path, oof_train.reshape(-1, 1), delimiter=',', fmt='%.5f')

# Export oof_test
file_path = os.path.join(OOF_PATH, MODEL_NAME + '_test.csv')
pd.DataFrame({MODEL_NAME: oof_test.reshape(-1, )}).to_csv(file_path, index=False)
# np.savetxt(file_path, oof_test.reshape(-1, 1), delimiter=',', fmt='%.5f')
print('SUCCESSFULLY SAVE {} AT {}  PLEASE VERIFY THEM'.format(MODEL_NAME, OOF_PATH))
