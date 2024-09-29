from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
import optuna

from utils import Imputer, MainFeatureExtractor, SimpleStatisticsExtractor, GeoFeatureExtractor, UserFeatureExtractor



path = "data/"
data_start = pd.read_csv(path + 'train_events.csv')
video_start = pd.read_csv(path + 'video_info_v2.csv')
targets_start = pd.read_csv(path + 'train_targets.csv')



data, features = Imputer().fit_transform(data_start, video_start, targets_start)

for extractor in [MainFeatureExtractor(), SimpleStatisticsExtractor(), GeoFeatureExtractor(), UserFeatureExtractor()]:
    data, features = extractor.fit_transform(data, features)

features.shape



class CatboostEstimator:
    """
    Класс для обучения Catboost
    """

    def fit(self, X, y, n_splits, cat_features, score, catboost_params):
        """
        Разбивает данные на k фолдов со стратификацией и обучает n_splits катбустов
        """
        self.one_model = False
        self.models = []
        scores = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for ind, (train_index, val_index) in enumerate(skf.split(X, y)):
            X_train = X.loc[train_index]
            y_train = y.loc[train_index]
            X_val = X.loc[val_index]
            y_val = y.loc[val_index]
            
            model = CatBoostClassifier(cat_features=cat_features, task_type="GPU", devices=['3', '4', '5', '6', '7'], **catboost_params)
            model.fit(X_train, y_train, verbose=500, eval_set=(X_val, y_val))
            
            self.models.append(model)
            y_pred = model.predict(X_val)
            scores.append(score(y_val, y_pred))
            print(f'model {ind}: score = {round(scores[-1], 4)}')
        
        scores = np.array(scores)
        print(f'mean score = {scores.mean().round(4)}, std = {scores.std().round(4)}')
        overall_score = scores.mean() - scores.std()
        print(f'overall score = {overall_score.round(4)}')
        return overall_score
            
    
    def fit_select_features(self, X, y, cat_features, to_drop):
        """
        Обучает один катбуст и выполняет elect features
        """
        self.one_model = True
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = CatBoostClassifier(cat_features=cat_features, verbose=150, iterations=2000)
        
        self.model.select_features(X_train, y_train, verbose=500, eval_set=(X_val, y_val), steps=10,
                                  num_features_to_select=30, features_for_select=X.columns,
                                  algorithm='RecursiveByLossFunctionChange', train_final_model=True)

    def predict(self, X, cnt_classes):
        if self.one_model:
            return self.model.predict_proba(X)
        
        y_pred = np.zeros((X.shape[0], cnt_classes))

        for model in self.models:
            y_pred += model.predict_proba(X)
        y_pred /= cnt_classes
        y_pred = np.argmax(y_pred, axis=1)
        
        return y_pred



from sklearn.metrics import f1_score, accuracy_score



def score_sex(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def score_age(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')



cat_features = []
for i, col in enumerate(features.columns):
    if features[col].dtype in ['object', 'category']:
        cat_features.append(col)
        
cat_features



def objective_sex(trial):
    params = {
        'iterations': 1500,  # trial.suggest_int('iterations', 300, 1200),
        'depth': trial.suggest_int('depth', 3, 8),
        'random_seed': 42,
        'verbose': 500,
    }

    catboost_sex = CatboostEstimator()
    
    features_to_drop = [
        'viewer_uid',
        'age'
    ]
    
    target_sex = 'sex'
    target_age = 'age_class'

    print('Sex model\n')
    
    score = catboost_sex.fit(
        X=features.drop(columns=features_to_drop + [target_sex] + [target_age]),
        y=features[target_sex],
        n_splits=2,
        cat_features=cat_features,
        score=score_sex,
        catboost_params=params,
    )
    
    return score



study = optuna.create_study(direction='maximize')
study.optimize(objective_sex, n_trials=40)



def objective_age(trial):
    params = {
        'iterations': 1500,  # trial.suggest_int('iterations', 300, 1200),
        'depth': trial.suggest_int('depth', 3, 8),
        'random_seed': 42,
        'verbose': 500,
    }

    print('\n\n\nAge model\n')
    
    catboost_age = CatboostEstimator()

    features_to_drop = [
        'viewer_uid',
        'age'
    ]

    target_sex = 'sex'
    target_age = 'age_class'

    print('Age model\n')

    score = catboost_age.fit(
        X=features.drop(columns=features_to_drop + [target_sex] + [target_age]),
        y=features[target_age],
        n_splits=2,
        cat_features=cat_features,
        score=score_age,
        catboost_params=params,
    )

    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective_age, n_trials=40)
