from utils import Imputer, MainFeatureExtractor, SimpleStatisticsExtractor, GeoFeatureExtractor, UserFeatureExtractor
from utils import BagOfWordsExtractor, TargetEncodingExtractor, TargetEncodingExtractorSex, ALSEmbeddingExtractor

from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
import utils
from utils.catboost_estimator import CatboostEstimator, score_sex, score_age
from sklearn.metrics import f1_score, accuracy_score


path = "data/"
data_start = pd.read_csv(path + 'train_events.csv')
video_start = pd.read_csv(path + 'video_info_v2.csv')
targets_start = pd.read_csv(path + 'train_targets.csv')

import warnings
warnings.filterwarnings('ignore')

data_start = data_start.truncate(after=100)

events, features = Imputer().fit_transform(data_start, video_start, targets_start)

# TODO: Generate ALS embeddings and save

for extractor in [MainFeatureExtractor(), ALSEmbeddingExtractor(), BagOfWordsExtractor(), SimpleStatisticsExtractor(), GeoFeatureExtractor(), UserFeatureExtractor()]:
    events, features = extractor.fit_transform(events, features)

print("Features were extracted")

cat_features = []
for i, col in enumerate(features.columns):
    if features[col].dtype in ['object', 'category']:
        cat_features.append(col)


catboost_sex = CatboostEstimator()
catboost_age = CatboostEstimator()

features_to_drop = [
    'age'
]

target_sex = 'sex'
target_age = 'age_class'


y_sex = features.reset_index()[[target_sex, 'viewer_uid']]
y_sex.name = target_sex

y_age = features.set_index('viewer_uid')[target_age]
y_age.name = target_age

print('Sex model\n')

catboost_sex.fit_with_features_selection(
    X=features.drop(columns=features_to_drop + [target_sex] + [target_age]),
    y=y_sex,
    events=events,
    cat_features=cat_features,
)

catboost_sex.model.save_model('catboost_sex')

print('\n\n\nAge model\n')

catboost_age.fit_with_features_selection(
    X=features.drop(columns=features_to_drop + [target_sex] + [target_age]),
    y=y_age,
    events=events,
    cat_features=cat_features,
)

catboost_age.model.save_model('catboost_age')
