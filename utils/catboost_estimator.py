from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import utils
import pandas as pd


class CatboostEstimator:
    """
    Класс для обучения Catboost
    """

    def fit(self, X, y, ids, events, n_splits, cat_features, score):
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

            train_idx = ids.loc[train_index]
            train_idx.name = 'viewer_uid'

            y_train_idx = y_train.copy()
            y_train_idx.index = train_idx
            X_train['viewer_uid'] = train_idx
            val_idx = ids.loc[val_index]
            val_idx.name = 'viewer_uid'
            X_val['viewer_uid'] = val_idx
            if y.name == 'age_class':
                target_enc_ext = utils.TargetEncodingExtractor()
            else:
                target_enc_ext = utils.TargetEncodingExtractorSex()
            events_filtered_train = pd.merge(events, train_idx, on='viewer_uid', how='inner')
            target_enc_ext.fit(events_filtered_train, pd.merge(X_train, y_train_idx, on='viewer_uid', how='inner'))
            events_filtered_test = pd.merge(events, val_idx, on='viewer_uid', how='inner')

            X_train = target_enc_ext.transform(events_filtered_train, X_train).drop(columns=['viewer_uid'])
            X_val = target_enc_ext.transform(events_filtered_test, X_val).drop(columns=['viewer_uid'])

            model = CatBoostClassifier(cat_features=cat_features, verbose=500, iterations=1000, depth=8, l2_leaf_reg=1.969)
            model.fit(X_train, y_train, verbose=500, eval_set=(X_val, y_val))

            self.models.append(model)
            y_pred = model.predict(X_val)
            scores.append(score(y_val, y_pred))
            print(f'model {ind}: score = {round(scores[-1], 4)}')

        scores = np.array(scores)
        print(f'mean score = {scores.mean().round(4)}, std = {scores.std().round(4)}')
        print(f'overall score = {(scores.mean() - scores.std()).round(4)}')


    def fit_with_features_selection(self, X, y, events, cat_features):
        """
        Обучает один катбуст и выполняет elect features
        """

        ids = X.reset_index()['viewer_uid']
        ids.name = 'viewer_uid'

        self.one_model = True

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        if y.name == 'age_class':
            target_enc_ext = utils.TargetEncodingExtractor()
        else:
            target_enc_ext = utils.TargetEncodingExtractorSex()
        events_filtered_train = pd.merge(events, y_train, on='viewer_uid', how='inner').drop(columns=[y.name])
        events_filtered_test = pd.merge(events, y_val, on='viewer_uid', how='inner').drop(columns=[y.name])

        target_enc_ext.fit(events_filtered_train, pd.merge(X_train, y_train, on='viewer_uid', how='inner'))

        X_train = target_enc_ext.transform(events_filtered_train, X_train).drop(columns=['viewer_uid'])
        X_val = target_enc_ext.transform(events_filtered_test, X_val).drop(columns=['viewer_uid'])
        y_train = y_train.drop(columns=['viewer_uid'])
        y_val = y_val.drop(columns=['viewer_uid'])

        self.model = CatBoostClassifier(cat_features=cat_features, verbose=150, iterations=2000, depth=8, l2_leaf_reg=1.969)

        self.model.select_features(X_train, y_train, verbose=500, eval_set=(X_val, y_val), steps=10,
                                   num_features_to_select=30, features_for_select=X_val.columns,
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




def score_sex(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def score_age(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')