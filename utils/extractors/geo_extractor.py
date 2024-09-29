from pathlib import Path

import pandas as pd
from math import log

path = Path(__file__).parent.absolute()

class GeoFeatureExtractor:
    """
    Класс для добавления фич

    Добавляет фичи
    - utc_delta: Разница в часах с UTC. Отображает удаленность региона от Москвы
    - region_user_count_log: Логарифм от количества пользователей в регионе
    """

    def fit(self):
        pass

    def transform(self, events, features):
        new_features = events.groupby('viewer_uid').agg(
            utc_delta=('utc_delta', lambda x: x.value_counts().idxmax()),
        )

        region_to_count_log = events.groupby('region').agg(
            region_user_count_log=('viewer_uid', lambda x: log(x.count())),
        )
        viewer_to_region = events.groupby('viewer_uid').agg(
            region=('region', lambda x: x.value_counts().idxmax()),
        )
        region_to_count_log = region_to_count_log.reset_index()
        viewer_to_region = viewer_to_region.reset_index()
        viewer_to_count = pd.merge(viewer_to_region, region_to_count_log, on='region', how='left')

        features = pd.merge(new_features, features, on='viewer_uid', how='inner')
        features = pd.merge(viewer_to_count, features, on='viewer_uid', how='right')

        region_counts = features['region'].value_counts()
        top_30_regions = region_counts.nlargest(30).index
        features['region'] = features['region'].where(features['region'].isin(top_30_regions), 'other')

        return events, features

    def fit_transform(self, events, features):
        print('Applying GeoFeatureExtractor...')
        self.fit()
        return self.transform(events, features)