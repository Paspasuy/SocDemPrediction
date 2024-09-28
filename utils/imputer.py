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
        # Оставляем только локальное время и разницу с utc в часах
        events = add_tz_and_localtime_column(events)
        events = events.drop('event_timestamp', axis=1)
        events = events.drop('timezone', axis=1)
        # Удаляем ненужный префикс video_
        events["rutube_video_id"] = events["rutube_video_id"].apply(lambda s: s[6:])
        video["rutube_video_id"] = video["rutube_video_id"].apply(lambda s: s[6:])

        # Мержим с информацией о роликах
        events = pd.merge(events, video, on='rutube_video_id', how='inner')

        return events, features

    def fit_transform(self, data, video, targets):
        self.fit()
        return self.transform(data, video, targets)
