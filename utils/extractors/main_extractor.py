from pathlib import Path

import pandas as pd

path = Path(__file__).parent.absolute()

class MainFeatureExtractor:
    """
    Класс для добавления фич

    Добавляет эмбеддинги пользователей, а также фичи favourite_cat и percent_fav_cat
    """
    def __init__(self):
        self.user_embed = None

    def fit(self, events, features):
        self.user_embed = pd.read_parquet(path.parent.parent / 'personal' / 'knifeman' / 'data' / 'target_embeds-custom_aggregation.parquet')

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