from pathlib import Path

import pandas as pd
import numpy as np

path = Path(__file__).parent.absolute()

class TargetEncodingExtractorSex:
    """
    Класс для добавления фич

    Обучается на таргетах, и использует эту информацию при transform.
    Например, сохраняется статистика по авторам, или по видео - какие категории людей их смотрят.

    Добавляет фичи:
    - targ_auth_sex_mean: Взвешенная сумма доли женского пола зрителей по авторам видеороликов
    - targ_vid_sex_mean{i}: Взвешенная сумма доли женского пола зрителей по видеороликам
    Для обеих фичей ставится -1, если информации не достаточно.
    """
    def __init__(self):
        self.authors_mean = None
        self.videos_mean = None
        self.allowed_auth = None
        self.allowed_vid = None

    def fit(self, events_train, features_train):
        """
        :param events_train
        :param features_train
        :return:
        """
        with_targets = pd.merge(events_train, features_train[['sex', 'viewer_uid']], on='viewer_uid', how='left')
        print(with_targets.head())
        self.authors_mean = with_targets.groupby('author_id').agg(
            authors_mean=('sex', 'mean'),
        )
        author_counts = events_train.groupby('author_id').size()
        threshold = author_counts.quantile(0.95)

        self.authors_mean = self.authors_mean.where(author_counts >= threshold, -100000)


        self.videos_mean = with_targets.groupby('rutube_video_id').agg(
            videos_mean=('sex', 'mean'),
        )

        video_counts = events_train.groupby('rutube_video_id').size()
        threshold = video_counts.quantile(0.95)

        self.videos_mean = self.videos_mean.where(video_counts >= threshold, -100000)

        self.allowed_auth = author_counts >= threshold
        self.allowed_auth.name = 'allowed_auth'
        self.allowed_vid = video_counts >= threshold
        self.allowed_vid.name = 'allowed_vid'


    def transform(self, events_test, features_test):
        events_test = events_test.copy()
        features_test = features_test.copy()

        user_duration_sum = events_test.groupby('viewer_uid').agg(
            user_duration_sum=('duration_sec', 'sum'),
        )
        events_with_sum = pd.merge(events_test, user_duration_sum, on='viewer_uid', how='left')
        events_test['event_weight'] = events_test['total_watchtime'] / events_with_sum['duration_sec']
        events_test = pd.merge(events_test, self.allowed_auth, on='author_id', how='left')
        events_test = pd.merge(events_test, self.allowed_vid, on='rutube_video_id', how='left')
        events_test['event_weight'] = events_test['event_weight'] * events_test['allowed_auth']
        events_test['event_weight'] = events_test['event_weight'] * events_test['allowed_vid']


        # Normalize weights so that for every user sum of weights is 1
        with_weight_sum = events_test.groupby('viewer_uid').agg(
            event_weight_sum=('event_weight', 'sum'),
        )
        events_with_weight_sum = pd.merge(events_test, with_weight_sum, on='viewer_uid', how='left')
        events_test['event_weight'] = events_test['event_weight'] / (events_with_weight_sum['event_weight_sum'].where(events_with_weight_sum.event_weight_sum > 0, 1))


        events_test = pd.merge(events_test, self.authors_mean, on='author_id', how='left')
        events_test = pd.merge(events_test, self.videos_mean, on='rutube_video_id', how='left')
        events_test['authors_mean'] = (events_test['authors_mean'] * events_test['event_weight']).fillna(-1)
        events_test['videos_mean'] = (events_test['videos_mean'] * events_test['event_weight']).fillna(-1)


        users_cats = events_test.groupby('viewer_uid').agg(
            targ_auth_sex_mean=('authors_mean', 'sum'),
            targ_vid_sex_mean=('videos_mean', 'sum'),
        )

        features_test = pd.merge(users_cats, features_test, on='viewer_uid', how='inner').fillna(0)

        return features_test
