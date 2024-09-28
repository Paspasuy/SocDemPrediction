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

    def transform(self, data, video, targets):
        data = data.copy()
        video = video.copy()
        targets = targets.copy()

        video['duration_sec'] = video['duration'] // 1000
        targets['sex'] = targets['sex'].apply(lambda x: 0 if x == 'male' else 1)
        # Оставляем только локальное время без таймзоны
        data = add_tz_and_localtime_column(data)
        data = data.drop('timezone', axis=1)
        data = data.drop('event_timestamp', axis=1)
        # Удаляем ненужный префикс video_
        data["rutube_video_id"].apply(lambda s: s[6:])
        video["rutube_video_id"].apply(lambda s: s[6:])

        return data, video, targets

    def fit_transform(self, data, video, targets):
        self.fit()
        return self.transform(data, video, targets)


class FeatureExtractor:
    """
    Класс для добавления фич
    """
    def __init__(self):
        self.user_embed = None

    def fit(self):
        self.user_embed = pd.read_parquet(path.parent / 'personal' / 'knifeman' / 'data' / 'target_embeds-custom_aggregation.parquet')

    def transform(self, data, video, targets):
        data = data.copy()
        video = video.copy()
        targets = targets.copy()

        data = pd.merge(data, video[['rutube_video_id', 'category']], on='rutube_video_id', how='inner')
        users_cats = data.groupby('viewer_uid').agg(
            favourite_cat=('category', lambda x: x.value_counts().idxmax()),
            percent_fav_cat=('category', lambda x: x.value_counts().max() / len(x))
        )
        # Add user embeddings from .parquet file
        users_cats = pd.merge(users_cats, self.user_embed, on='viewer_uid', how='inner')

        targets = pd.merge(users_cats, targets, on='viewer_uid', how='inner')

        return data, video, targets

    def fit_transform(self, data, video, targets):
        self.fit()
        return self.transform(data, video, targets)