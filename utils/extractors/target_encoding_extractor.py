from pathlib import Path

import pandas as pd
import numpy as np

path = Path(__file__).parent.absolute()

class TargetEncodingExtractor:
    """
    Класс для добавления фич

    Добавляет
    """
    def __init__(self):
        self.authors_categories = None

    def fit(self, events_train, features_train):
        """
        :param events_train:
        :param features_train: Содержат также таргеты
        :return:
        """
        with_targets = pd.merge(events_train, features_train[['age_class', 'viewer_uid']], on='viewer_uid', how='left')
        with_targets = pd.get_dummies(with_targets, columns=['age_class'])
        self.authors_categories = with_targets.groupby('author_id').agg(
            age_class_0=('age_class_0', np.sum),
            age_class_1=('age_class_1', np.sum),
            age_class_2=('age_class_2', np.sum),
            age_class_3=('age_class_3', np.sum),
        )
        self.authors_categories = self.authors_categories.div(self.authors_categories.sum(axis=1), axis=0)


    def transform(self, events_test, features_test):
        events_test = events_test.copy()

        user_duration_sum = events_test.groupby('viewer_uid').agg(
            user_duration_sum=('duration_sec', np.sum),
        )
        events_with_sum = pd.merge(events_test, user_duration_sum, on='viewer_uid', how='left')
        events_test['event_weight'] = events_test['total_watchtime'] / events_with_sum['duration_sec']

        # Normalize weights so that for every user sum of weights is 1
        with_weight_sum = events_test.groupby('viewer_uid').agg(
            event_weight_sum=('event_weight', np.sum),
        )
        events_with_weight_sum = pd.merge(events_test, with_weight_sum, on='viewer_uid', how='left')
        events_test['event_weight'] = events_test['event_weight'] / events_with_weight_sum['event_weight_sum']

        events_test = pd.merge(events_test, self.authors_categories, on='author_id', how='left')
        for i in range(4):
            events_test[f'age_class_{i}'] = events_test[f'age_class_{i}'] * events_test['event_weight']

        users_cats = events_test.groupby('viewer_uid').agg(
            age_class_0=('age_class_0', np.sum),
            age_class_1=('age_class_1', np.sum),
            age_class_2=('age_class_2', np.sum),
            age_class_3=('age_class_3', np.sum),
        )

        features = pd.merge(users_cats, features_test, on='viewer_uid', how='inner')

        return features
