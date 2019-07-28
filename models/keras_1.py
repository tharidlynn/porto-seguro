
# coding: utf-8

# https://www.kaggle.com/tilii7/keras-averaging-runs-gini-early-stopping


from utils import utils, gini
import time
from constants import *
import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold

from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, CSVLogger
from keras.wrappers.scikit_learn import KerasClassifier

MODEL_NAME = 'keras_1'

combined = utils.load_data()
combined = utils.cat_transform(combined, 'onehot')

train, test = utils.recover_train_test_na(combined, fillna=True)

# Fillna for minmax scaler
train = train.replace(np.NaN, -1)
test = test.replace(np.NaN, -1)

X_train = train.drop('target', axis=1)
y_train = train.target
X_test = test

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test =  scaler.transform(X_test)


class gini_callback(Callback):
    def __init__(self, training_data, validation_data):
        self.X_tr = training_data[0]
        self.y_tr = training_data[1]
        self.X_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_tr = self.model.predict_proba(self.X_tr)
#         roc = roc_auc_score(self.y_tr, y_pred_tr)
#         logs['roc_auc'] = roc
#         logs['gini_tr'] = (roc * 2 ) - 1
        logs['gini_tr'] = gini.gini_sklearn(self.y_tr, y_pred_tr)

        y_pred_val = self.model.predict_proba(self.X_val)
#         roc = roc_auc_score(self.y_val, y_pred_val)
#         logs['roc_auc_val'] = roc
#         logs['gini_val'] = (roc * 2 ) - 1
        logs['gini_val'] = gini.gini_sklearn(self.y_val, y_pred_val)


        print('Gini Score in training set: {},  test set: {}'.format(logs['gini_tr'], logs['gini_val']))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


log_path = os.path.join(LOG_PATH, MODEL_NAME + '_log.csv')
checkpoint_path = os.path.join(LOG_PATH, MODEL_NAME + '_check.check')


epochs = 128
batch_size = 128
patience = 10


tmp = time.time()
skf = StratifiedKFold(n_splits=KFOLDS, random_state=SEED)
scores = []
# rows X cols
oof_train = np.zeros((X_train.shape[0], 1))
oof_test = np.zeros((X_test.shape[0], 1))

for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
    assert len(X_train) == len(y_train)

    print('\n')

    print('[Fold {}/{} START]'.format(i + 1, KFOLDS))

    X_tr, X_val = X_train[train_index], X_train[val_index]
    y_tr, y_val = y_train[train_index], y_train[val_index]

    def create_model():
        model = Sequential()
        model.add(
            Dense(
                256,
                input_dim=X_tr.shape[1],
                kernel_initializer='glorot_normal',
                ))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(128, kernel_initializer='glorot_normal'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))
        model.add(Dense(64, kernel_initializer='glorot_normal'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.15))
        model.add(Dense(32, kernel_initializer='glorot_normal'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='sigmoid'))

        # Compile model
    #     optimizer = SGD(lr=learn_rate, momentum=momentum)

        model.compile(optimizer='sgd', metrics=['accuracy'], loss='binary_crossentropy')

        return model


    callbacks = [
    gini_callback(training_data=(X_tr, y_tr), validation_data=(X_val, y_val)),
    EarlyStopping(monitor='gini_val', patience=patience, mode='max', verbose=1),
    CSVLogger(log_path, separator=',', append=False),
    ModelCheckpoint(checkpoint_path, monitor='gini_val', mode='max', save_best_only=True, verbose=1)
]
    model = KerasClassifier(build_fn=create_model,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=99,
                        callbacks=callbacks)

    model.fit(X_tr, y_tr)

    # We want the best model from checkpoint
    del model
    model = load_model(checkpoint_path)

    # For train and valid only
    pred_val = model.predict_proba(X_val)
    oof_train[val_index] = pred_val

    # Store average score for evaluate model
    scores.append(gini.gini_sklearn(y_val, pred_val))

    # Store test predictions for submissions
    pred_test = model.predict_proba(X_test)[:,1] / KFOLDS
    oof_test += pred_test

    print('[Fold {}/{} Gini score: {}]'.format(i+1, KFOLDS, scores[i]))

    gc.collect()
    print('[Fold {}/{} END]'.format(i+1, KFOLDS))

print('Average score: {}'.format(np.mean(scores)))
print('Total run time: {} seconds'.format(time.time() - tmp))

# Export oof_train
file_path = os.path.join(OOF_PATH, MODEL_NAME + '_train.csv')
pd.DataFrame({MODEL_NAME: oof_train}).to_csv(file_path, index=False)
# np.savetxt(file_path, oof_train.reshape(-1, 1), delimiter=',', fmt='%.5f')

# Export oof_test
file_path = os.path.join(OOF_PATH, MODEL_NAME + '_test.csv')
pd.DataFrame({MODEL_NAME: oof_test}).to_csv(file_path, index=False)
# np.savetxt(file_path, oof_test.reshape(-1, 1), delimiter=',', fmt='%.5f')
print('SUCCESSFULLY SAVE {} AT {}  PLEASE VERIFY THEM'.format(MODEL_NAME, OOF_PATH))
