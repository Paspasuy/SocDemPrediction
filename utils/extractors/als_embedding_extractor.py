from pathlib import Path

import pandas as pd

path = Path(__file__).parent.absolute()

class ALSEmbeddingExtractor:
    """
    Класс для добавления фич

    Добавляет эмбеддинги пользователей полученные из ALS в ноутуке als.ipynb
    """
    def __init__(self):
        self.user_embed = None

    def fit(self, events, features):
        try:
            self.user_embed = pd.read_parquet(path.parent.parent / 'personal' / 'flypew' / 'data' / 'user_embeddings.parquet')
        except FileNotFoundError as e:
            print("ALS embeddings not found, skipping the extractor...")
            self.user_embed = None

    def transform(self, events, features):
        if self.user_embed is None:
            return events, features
        print('Applying ALSEmbeddingExtractor...')
        features = features.copy()

        features = pd.merge(features, self.user_embed, on='viewer_uid', how='left').fillna(0)

        return events, features

    def fit_transform(self, events, features):
        self.fit(events, features)
        return self.transform(events, features)