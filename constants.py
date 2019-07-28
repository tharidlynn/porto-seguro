import os

# MODEL_SEED
MODEL_SEED = 22

# KF_SEED
KF_SEED = 100

# KFOLDS
KFOLDS = 5

# Root directory
PROJECT_PATH = os.getcwd()

# Data
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
DATA_TRAIN_PATH = os.path.join(DATA_PATH, 'train.csv')
DATA_TEST_PATH = os.path.join(DATA_PATH, 'test.csv')

# Models
MODELS_PATH = os.path.join(PROJECT_PATH, 'models')

# Output for oof
OOF_PATH = os.path.join(PROJECT_PATH, 'oof')
# Output for submission
SUBMISSION_PATH = os.path.join(PROJECT_PATH, 'submissions')

# Log file
LOG_PATH = os.path.join(PROJECT_PATH, 'logs')
