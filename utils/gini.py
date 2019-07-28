# Define the gini metric - from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897
import numpy as np

def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

# Create an XGBoost-native core metric from Gini
def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return [('gini', gini_score)]

# Create an lgbm-native core metric from Gini
def gini_lgb(preds, dtrain):
    labels = list(dtrain.get_label())
    gini_score = gini_normalized(labels, preds)
    return [('gini', gini_score, True)]

# Use in scikitlearn
def gini_sklearn(y_true, y_pred):
    gini_score = gini_normalized(y_true, y_pred)
    return gini_score

# Use in xgb scikitlearn wrapper
def gini_xgbsklearn(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return [('gini',-1 * gini_score)]

# Use in lgbm scikitlearn wrapper
def gini_lgbsklearn(y_true, preds):
    gini_score = gini_normalized(y_true, preds)
    return [('gini', gini_score, True)]

def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini
