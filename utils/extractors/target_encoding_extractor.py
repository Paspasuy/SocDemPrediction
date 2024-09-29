from pathlib import Path

import pandas as pd
import numpy as np

path = Path(__file__).parent.absolute()

class TargetEncodingExtractor:
    """
    Класс для добавления фич

    Обучается на таргетах, и использует эту информацию при transform.
    Например, сохраняется статистика по авторам, или по видео - какие категории людей их смотрят.

    Добавляет фичи:
    - targ_auth_age_class_{i}: Взвешенная сумма распределений зрителей по авторам видеороликов
    - targ_vid_age_class_{i}: Взвешенная сумма распределений зрителей по видеороликам
    """
    def __init__(self):
        self.authors_categories = None
        self.videos_categories = None

    def fit(self, events_train, features_train):
        """
        :param events_train
        :param features_train
        :return:
        """
        with_targets = pd.merge(events_train, features_train[['age_class', 'viewer_uid']], on='viewer_uid', how='left')
        with_targets = pd.get_dummies(with_targets, columns=['age_class'])



        self.authors_categories = with_targets.groupby('author_id').agg(
            targ_auth_age_class_0=('age_class_0', 'sum'),
            targ_auth_age_class_1=('age_class_1', 'sum'),
            targ_auth_age_class_2=('age_class_2', 'sum'),
            targ_auth_age_class_3=('age_class_3', 'sum'),
        )
        self.authors_categories = self.authors_categories.div(self.authors_categories.sum(axis=1), axis=0)

        author_counts = events_train.groupby('author_id').size()
        threshold = author_counts.quantile(0.95)

        for i in range(4):
            col = f'targ_auth_age_class_{i}'
            self.authors_categories[col] = self.authors_categories[col].where(author_counts >= threshold, 0)



        self.videos_categories = with_targets.groupby('rutube_video_id').agg(
            targ_vid_age_class_0=('age_class_0', 'sum'),
            targ_vid_age_class_1=('age_class_1', 'sum'),
            targ_vid_age_class_2=('age_class_2', 'sum'),
            targ_vid_age_class_3=('age_class_3', 'sum'),
        )
        self.videos_categories = self.videos_categories.div(self.videos_categories.sum(axis=1), axis=0)

        video_counts = events_train.groupby('rutube_video_id').size()
        threshold = video_counts.quantile(0.95)

        for i in range(4):
            col = f'targ_vid_age_class_{i}'
            self.videos_categories[col] = self.videos_categories[col].where(video_counts >= threshold, 0)


    def transform(self, events_test, features_test):
        events_test = events_test.copy()
        features_test = features_test.copy()

        user_duration_sum = events_test.groupby('viewer_uid').agg(
            user_duration_sum=('duration_sec', 'sum'),
        )
        events_with_sum = pd.merge(events_test, user_duration_sum, on='viewer_uid', how='left')
        events_test['event_weight'] = events_test['total_watchtime'] / events_with_sum['duration_sec']

        # Normalize weights so that for every user sum of weights is 1
        with_weight_sum = events_test.groupby('viewer_uid').agg(
            event_weight_sum=('event_weight', 'sum'),
        )
        events_with_weight_sum = pd.merge(events_test, with_weight_sum, on='viewer_uid', how='left')
        events_test['event_weight'] = events_test['event_weight'] / events_with_weight_sum['event_weight_sum']



        events_test = pd.merge(events_test, self.authors_categories, on='author_id', how='left')
        for i in range(4):
            col = f'targ_auth_age_class_{i}'
            events_test[col] = events_test[col] * events_test['event_weight']


        events_test = pd.merge(events_test, self.videos_categories, on='rutube_video_id', how='left')
        for i in range(4):
            col = f'targ_vid_age_class_{i}'
            events_test[col] = events_test[col] * events_test['event_weight']


        users_cats_auth = events_test.groupby('viewer_uid').agg(
            targ_auth_age_class_0=('targ_auth_age_class_0', 'sum'),
            targ_auth_age_class_1=('targ_auth_age_class_1', 'sum'),
            targ_auth_age_class_2=('targ_auth_age_class_2', 'sum'),
            targ_auth_age_class_3=('targ_auth_age_class_3', 'sum'),
        )

        users_cats_vid = events_test.groupby('viewer_uid').agg(
            targ_vid_age_class_0=('targ_vid_age_class_0', 'sum'),
            targ_vid_age_class_1=('targ_vid_age_class_1', 'sum'),
            targ_vid_age_class_2=('targ_vid_age_class_2', 'sum'),
            targ_vid_age_class_3=('targ_vid_age_class_3', 'sum'),
        )
        users_cats_auth = users_cats_auth.div(users_cats_auth.sum(axis=1), axis=0)
        users_cats_vid = users_cats_vid.div(users_cats_vid.sum(axis=1), axis=0)


        features_test = pd.merge(users_cats_auth, features_test, on='viewer_uid', how='inner').fillna(0)
        features_test = pd.merge(users_cats_vid, features_test, on='viewer_uid', how='inner').fillna(0)

        return features_test
