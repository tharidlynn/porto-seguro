from params import *
from utils import layer1

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier
from rgf.sklearn import RGFClassifier

from sklearn.linear_model import LogisticRegression

# from sklearn.linear_model import LogisticRegression
# from fastFM import sgd, als

# xgb_4 = layer1.Layer1Train(XGBClassifier(**xgb_4_params), 'xgb_4',
#         drop_stupid='default', data_transform='round', cat_transform='onehot')
#
# xgb_5 = layer1.Layer1Train(XGBClassifier(**xgb_5_params), 'xgb_5')
#
# xgb_6 = layer1.Layer1Train(XGBClassifier(**xgb_6_params), 'xgb_6',
#         drop_stupid='default', cat_transform='onehot', feature_interactions=True, engineer_stats=True)
#
# xgb_7 = layer1.Layer1Train(XGBClassifier(**xgb_7_params), 'xgb_7',
#         drop_stupid='default', cat_transform='onehot', feature_interactions=False, engineer_stats=True)

# xgb_8 = layer1.Layer1Train(XGBClassifier(**xgb_8_params), 'xgb_8',
#         kinetic_transform=False, drop_stupid='default', cat_transform='smooth', feature_interactions=False, recon_category=True, engineer_stats=True)

# xgb_9 = layer1.Layer1Train(XGBClassifier(**xgb_9_params), 'xgb_9',
#         drop_stupid='default', cat_transform='onehot', feature_interactions=False, recon_category=True, engineer_stats=True)
#
# xgb_10 = layer1.Layer1Train(XGBClassifier(**xgb_10_params), 'xgb_10',
#         drop_stupid='default', cat_transform='onehot', feature_interactions=True, recon_category=True, engineer_stats=True)

xgb_depth_10_itt = layer1.Layer1Train(XGBClassifier(**xgb_depth_10_itt_params), 'xgb_depth_10_itt',
        drop_stupid='default', cat_transform=False, recon_category=True, engineer_stats=True)

xgb_homeless_bag_kf = layer1.Layer1Train(XGBClassifier(**xgb_homeless_params), 'xgb_homeless_bag_kf',
        drop_stupid='default', cat_transform='smooth',  recon_category=True, engineer_stats=True)

xgb_18 = layer1.Layer1Train(XGBClassifier(**xgb_18_params), 'xgb_18',
         drop_stupid='default', cat_transform='smooth', recon_category=True, data_transform='round', engineer_stats=True)

rgf_bojan_na_bag_kf = layer1.Layer1Train(RGFClassifier(**rgf_bojan_params), 'rgf_bojan_na_bag_kf',
        drop_stupid='default', cat_transform='smooth', recon_category=True, engineer_stats=True)

# fm_sgd = layer1.Layer1Train(sgd.FMClassification(**fm_sgd_params), 'fm_sgd',
#         drop_stupid=True, cat_transform='onehot', recon_category=True, engineer_stats=True)
#
# fm_als = layer1.Layer1Train(sgd.FMClassification(**fm_als_params), 'fm_als',
#         drop_stupid=True, cat_transform='smooth', data_transform='log', recon_category=True, engineer_stats=True)

cat_1 = layer1.Layer1Train(CatBoostClassifier(**cat_1_params), 'cat_1',
        drop_stupid='default', recon_category=True, cat_transform=False, engineer_stats=True)

lgbm_1 = layer1.Layer1Train(LGBMClassifier(**lgbm_1_params), 'lgbm_1',
        drop_stupid='default', recon_category=True, cat_transform='smooth', data_transform='round', engineer_stats=True)

xgb_99 = layer1.Layer1Train(XGBClassifier(**xgb_99_params), 'xgb_99',
         drop_stupid='default', cat_transform='smooth', recon_category=True, engineer_stats=True, feature_interactions=False)

logistic = layer1.Layer1Train(LogisticRegression(**logistic_params), 'logistic',
        drop_stupid='default', cat_transform='onehot', recon_category=True, engineer_stats=True)

xgb_homeless_100 = layer1.Layer1Train(XGBClassifier(**xgb_homeless_params), 'xgb_homeless_100',
        drop_stupid='default', cat_transform='smooth',  recon_category=True, engineer_stats=True)

models = [xgb_homeless_100]

for model in models:
    model.train()
