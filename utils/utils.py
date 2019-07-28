from constants import *

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")


# Use this to load data
def load_data(path_train=DATA_TRAIN_PATH, path_test=DATA_TEST_PATH):
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)

    combined = pd.concat([train.drop(['target', 'id'], axis=1), test.drop('id', axis=1)], axis=0)
    print('Successfully Load and combined data')
    return combined

def drop_stupid(combined, type='default'):
    '''
    kak >>> ps_car_11_cat ps_car_10_cat
    ps_car 1 -9 ?
    car 11 ???
    ind 16 17 18 bin?
    '''
    if type == 'default':

        combined = combined.drop(combined.columns[combined.columns.str.startswith('ps_calc') == True], axis=1)

        # Drop this one because ind14 is sum of them
        combined = combined.drop(['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin'], axis=1)

        # Drop this one and you will gain a lot of scores
        combined = combined.drop(['ps_car_10_cat','ps_car_11_cat'], axis=1)

    elif type == 'olivier':
        olivier_features = [
            "ps_car_13",  #            : 1571.65 / shadow  609.23
            "ps_reg_03",  #            : 1408.42 / shadow  511.15
        	"ps_ind_05_cat",  #        : 1387.87 / shadow   84.72
        	"ps_ind_03",  #            : 1219.47 / shadow  230.55
        	"ps_ind_15",  #            :  922.18 / shadow  242.00
        	"ps_reg_02",  #            :  920.65 / shadow  267.50
        	"ps_car_14",  #            :  798.48 / shadow  549.58
        	"ps_car_12",  #            :  731.93 / shadow  293.62
        	"ps_car_01_cat",  #        :  698.07 / shadow  178.72
        	"ps_car_07_cat",  #        :  694.53 / shadow   36.35
        	"ps_ind_17_bin",  #        :  620.77 / shadow   23.15
        	"ps_car_03_cat",  #        :  611.73 / shadow   50.67
        	"ps_reg_01",  #            :  598.60 / shadow  178.57
        	"ps_car_15",  #            :  593.35 / shadow  226.43
        	"ps_ind_01",  #            :  547.32 / shadow  154.58
        	"ps_ind_16_bin",  #        :  475.37 / shadow   34.17
        	"ps_ind_07_bin",  #        :  435.28 / shadow   28.92
        	"ps_car_06_cat",  #        :  398.02 / shadow  212.43
        	"ps_car_04_cat",  #        :  376.87 / shadow   76.98
        	"ps_ind_06_bin",  #        :  370.97 / shadow   36.13
        	"ps_car_09_cat",  #        :  214.12 / shadow   81.38
        	"ps_car_02_cat",  #        :  203.03 / shadow   26.67
        	"ps_ind_02_cat",  #        :  189.47 / shadow   65.68
        	"ps_car_11",  #            :  173.28 / shadow   76.45
        	"ps_car_05_cat",  #        :  172.75 / shadow   62.92
        	"ps_calc_09",  #           :  169.13 / shadow  129.72
        	"ps_calc_05",  #           :  148.83 / shadow  120.68
        	"ps_ind_08_bin",  #        :  140.73 / shadow   27.63
        	"ps_car_08_cat",  #        :  120.87 / shadow   28.82
        	"ps_ind_09_bin",  #        :  113.92 / shadow   27.05
        	"ps_ind_04_cat",  #        :  107.27 / shadow   37.43
        	"ps_ind_18_bin",  #        :   77.42 / shadow   25.97
        	"ps_ind_12_bin",  #        :   39.67 / shadow   15.52
        	"ps_ind_14"  #            :   37.37 / shadow   16.65
        ]

        combined = combined[olivier_features]
    print('Successfully drop stupid with {}'.format(type))
    return combined

def cat_transform(combined, type):
    cat_col = combined.columns[combined.columns.str.endswith('cat') == True]
    if type == 'onehot':
        combined = pd.get_dummies(combined, columns=combined.columns[combined.columns.str.endswith('cat')==True])
    elif type == 'count':
        for col in cat_col:
            col_map = combined[col].value_counts()
            combined[str(col) + '_count'] = combined[col].map(col_map)
    elif type == 'mean':
        train, test = recover_train_test_na(combined, fillna=False)
        target = train.target
        mean_encoder = Bayesian_Encoding(nfolds=5, mode='likelihood')
        encoded_train, encoded_test = mean_encoder.fit_transform(train[cat_col], test[cat_col], target)
        for col in cat_col:
            train[col] = encoded_train[col]
            test[col] = encoded_test[col]
        combined = pd.concat([train.drop('target', axis=1), test], axis=0)
    print('Successfully category transform with {}'.format(type))
    return combined

def recon_category(combined):
    combined['ps_ind_0609_cat'] = np.zeros_like(combined.ps_ind_06_bin)
    combined['ps_ind_0609_cat'][combined.ps_ind_06_bin==1] = 1
    combined['ps_ind_0609_cat'][combined.ps_ind_07_bin==1] = 2
    combined['ps_ind_0609_cat'][combined.ps_ind_08_bin==1] = 3
    combined['ps_ind_0609_cat'][combined.ps_ind_09_bin==1] = 4
    combined['ps_ind_0609_cat'][combined.ps_ind_0609_cat==0] = 5

    combined.drop(combined.loc[:,'ps_ind_06_bin':'ps_ind_09_bin'].columns, axis=1, inplace=True)
    print('Successfully recon category')
    return combined


def recon(reg):
    integer = int(np.round((40*reg)**2))
    for a in range(32):
        if (integer - a) % 31 == 0:
            A = a
    M = (integer - A)//31
    return A, M

def feature_interactions(combined):

    copy_combined = load_data()
    combined_car = copy_combined[copy_combined.columns[copy_combined.columns.str.startswith('ps_car') == True]]
    combined_ind = copy_combined[copy_combined.columns[copy_combined.columns.str.startswith('ps_ind') == True]]
    combined_reg = copy_combined[copy_combined.columns[copy_combined.columns.str.startswith('ps_reg') == True]]

    # Some non-linear features
    # combined['ps_car_13_x_ps_reg_03'] = combined_car['ps_car_13'] * combined_reg['ps_reg_03']
    # combined['multi_reg'] = combined_reg['ps_reg_01'] * combined_reg['ps_reg_03'] * combined_reg['ps_reg_02']
    # combined['sum_reg'] = combined_reg['ps_reg_01'] + combined_reg['ps_reg_03'] + combined_reg['ps_reg_02']

    # combined['ps_car'] = combined_car['ps_car_13'] * combined_reg['ps_reg_03'] * combined_car['ps_car_13']
    # combined['ps_ind'] = combined_ind['ps_ind_03'] * combined_ind['ps_ind_15']
    #
    # combined['ps_reg_A'] = combined_reg['ps_reg_03'].apply(lambda x: recon(x)[0])
    # combined['ps_reg_M'] = combined_reg['ps_reg_03'].apply(lambda x: recon(x)[1])
    # combined['ps_reg_A'].replace(19,-1, inplace=True)
    # combined['ps_reg_M'].replace(51,-1, inplace=True)

    print('Successfully make some great feature interactions')

    return combined


def engineer_stats(combined):

    # Car 10 cat and car 11 cat
    copy_combined = load_data()
    combined_car = copy_combined[copy_combined.columns[copy_combined.columns.str.startswith('ps_car') == True]]
    combined_ind = copy_combined[copy_combined.columns[copy_combined.columns.str.startswith('ps_ind') == True]]
    combined_reg = copy_combined[copy_combined.columns[copy_combined.columns.str.startswith('ps_reg') == True]]

    combined['row_na'] = (copy_combined == -1).sum(axis=1)
    combined['count_car_na'] = (combined_car == -1).sum(axis=1)
    combined['count_car_zero'] = (combined_car == 0).sum(axis=1)
    combined['count_car_one'] = (combined_car == 1).sum(axis=1)
    combined['count_ind_na'] = (combined_ind == -1).sum(axis=1)
    combined['count_ind_zero'] = (combined_ind == 0).sum(axis=1)
    combined['count_ind_one'] = (combined_ind == 1).sum(axis=1)
    combined['count_reg_na'] = (combined_reg == -1).sum(axis=1)

    print('Successfully engineer stats')
    return combined


def recover_train_test_na(combined, fillna=True, remove_outliers=False):
    if fillna==True:
        # Fill Na
        combined.replace(-1, np.NaN, inplace=True)

    # Recover train
    targets = pd.read_csv(DATA_TRAIN_PATH).target.values
    train = combined.iloc[0:595212, :]
    if remove_outliers == True:
        train = outlier(train)
    train['target'] = targets
    # Recover test set
    test = combined.iloc[595212:]

    return train, test

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None,    # Revised to encode validation series
                  val_series=None,
                  tst_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_val_series = pd.merge(
        val_series.to_frame(val_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=val_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_val_series.index = val_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index

    return add_noise(ft_trn_series, noise_level), add_noise(ft_val_series, noise_level), add_noise(ft_tst_series, noise_level)

class Bayesian_Encoding(object):
    ''' mode can be a str of 'likelihood', 'weight_of_evidence', 'count', 'diff' '''
    def __init__(self, nfolds=5, mode = 'likelihood'):
        self.nfolds = nfolds
        self.mode = mode

    def encoder(self, train, col):
        if self.mode == 'likelihood':
            encoded = train.groupby(col).target.mean()
        elif self.mode == 'weight_of_evidence':
            target = train.groupby(col).target.sum()
            non_target = train.groupby(col).target.count()-target
            encoded = np.log(target/non_target)*100
        elif self.mode == 'count':
            encoded = train.groupby(col).target.sum()
        elif self.mode == 'diff':
            target = train.groupby(col).target.sum()
            non_target = train.groupby(col).target.count()-target
            encoded = target-non_target
        else:
            print('Error!! Please specify encoding mode')
        return encoded

    def cv_encoder(self, train, val, cols):
        for col in cols:
            target_mean = self.encoder(train, col)
            val[col] = val[col].map(target_mean)
        return val

    def global_mean(self, train):
        mean = pd.Series(np.zeros(train.columns.shape), index=train.columns)
        for col in train.columns:
            mean[col] = self.encoder(train, col).mean()
        return mean

    def fit_transform(self, train, test, target):
        train = pd.concat([train, target], axis=1)
        X_train = train.drop('target', axis=1)
        X_test = test
        cols = X_train.columns
        encoded_train = pd.DataFrame(np.zeros(X_train.shape), columns=X_train.columns)
        encoded_test = pd.DataFrame(np.zeros(X_test.shape), columns=X_test.columns)


        skf = StratifiedKFold(n_splits=self.nfolds, shuffle=False)
        for i, (train_index, val_index) in enumerate(skf.split(train, train.target)):
            print('[START ENCODING Fold {}/{}]'.format(i + 1, self.nfolds))
            X_tr, X_val = train.iloc[train_index,:], train.iloc[val_index,:]
            encoded_train.iloc[val_index,:] = self.cv_encoder(X_tr, X_val, cols)
            encoded_test += self.cv_encoder(X_tr, X_test, cols)/5

        # fill NA of encoded using Global Mean
        encoded_train = encoded_train.fillna(self.global_mean(train))
        encoded_test = encoded_test.fillna(self.global_mean(train))
        print('Successfully Encoded')
        return encoded_train, encoded_test

def outlier(train):
    num_cols = ['ps_reg_03', 'ps_car_12', 'ps_car_13', 'ps_car_14']
    for i in num_cols:
        quartile_1,quartile_3 = np.percentile(train[i],[25,75])
        quartile_f,quartile_l = np.percentile(train[i],[1,99])
        IQR = quartile_3-quartile_1
        lower_bound = quartile_1 - (1.5*IQR)
        upper_bound = quartile_3 + (1.5*IQR)
        print(lower_bound,upper_bound)
        print(quartile_f,quartile_l)

        train[i].loc[train[i] < lower_bound] = quartile_f
        train[i].loc[train[i] > upper_bound] = quartile_l
    print('Successfully remove outliers in num_cols')
    return train


def kinetic(row):
        probs=np.unique(row,return_counts=True)[1]/len(row)
        kinetic=np.sum(probs**2)
        return kinetic

def kinetic_transform(combined):
    kin_ind = combined[combined.columns[combined.columns.str.contains('ind')]]
    kin_car = combined[combined.columns[combined.columns.str.contains('car') & combined.columns.str.endswith('cat')]]
    kin_calc_not_bin = combined[combined.columns[combined.columns.str.contains('calc') & ~(combined.columns.str.contains('bin'))]]
    kin_calc_bin = combined[combined.columns[combined.columns.str.contains('calc') & combined.columns.str.contains('bin')]]
    kin_arr = [kin_ind, kin_car, kin_calc_not_bin, kin_calc_bin]

    for i, kin in enumerate(kin_arr):
        combined['kin_{}'.format(i+1)] = kin.apply(kinetic, axis=1)

    print ('Transform kinetic features successfully')
    return combined

def data_transform(combined, type):
    print(type, 'has been selected')
    if type == 'log':
        combined = combined.apply(np.log1p)
    elif type == 'round':
        combined = combined.round(2)
    elif type == 'power':
        combined = combined.apply(lambda x: x**2)
    elif type == 'sqrt':
        combined = combined.apply(np.sqrt)
    elif type == 'minmax':
        scaler = MinMaxScaler()
        combined = scaler.fit_transform(combined)
    elif type == 'std':
        scaler = StandardScaler()
        combined = scaler.fit_transform(combined)
    elif type == 'pca':
        pass
    elif type == 'tsne':
        scaler = StandardScaler()
        scale_combined = scaler.fit_transform(combined)

        tsne = TSNE()
        combined = tsne.fit_transform(scale_combined)
    else:
        print('ERROR NO TYPE !!!')
    print('Successfully data transform with {}'.format(type))
    return combined
