from pathlib import Path

import pandas as pd

from utils import add_tz_and_localtime_column

path = Path(__file__).parent.absolute()


class Imputer:
    """
    Класс для приведения датасета к нормальной форме
    """

    def fit(self):
        pass

    def transform(self, events, video, features):
        events = events.copy()
        video = video.copy()
        features = features.copy()

        video['duration_sec'] = video['duration'] // 1000
        features['sex'] = features['sex'].apply(lambda x: 0 if x == 'male' else 1)
        # Оставляем только локальное время без таймзоны
        events = add_tz_and_localtime_column(events)
        events = events.drop('timezone', axis=1)
        events = events.drop('event_timestamp', axis=1)
        # Удаляем ненужный префикс video_
        events["rutube_video_id"] = events["rutube_video_id"].apply(lambda s: s[6:])
        video["rutube_video_id"] = video["rutube_video_id"].apply(lambda s: s[6:])

        # Мержим с информацией о роликах
        events = pd.merge(events, video, on='rutube_video_id', how='inner')

        return events, features

    def fit_transform(self, data, video, targets):
        self.fit()
        return self.transform(data, video, targets)


class FeatureExtractor:
    """
    Класс для добавления фич
    """
    def __init__(self):
        self.user_embed = None

    def fit(self, events, features):
        self.user_embed = pd.read_parquet(path.parent / 'personal' / 'knifeman' / 'data' / 'target_embeds-custom_aggregation.parquet')

    def transform(self, events, features):
        events = events.copy()
        features = features.copy()

        users_cats = events.groupby('viewer_uid').agg(
            favourite_cat=('category', lambda x: x.value_counts().idxmax()),
            percent_fav_cat=('category', lambda x: x.value_counts().max() / len(x))
        )
        # Add user embeddings from .parquet file
        users_cats = pd.merge(users_cats, self.user_embed, on='viewer_uid', how='inner')

        features = pd.merge(users_cats, features, on='viewer_uid', how='inner')

        return events, features

    def fit_transform(self, events, features):
        self.fit(events, features)
        return self.transform(events, features)