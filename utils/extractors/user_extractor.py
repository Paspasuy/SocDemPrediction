from pathlib import Path

import pandas as pd
from math import log

path = Path(__file__).parent.absolute()


def _get_get_dev(device):
    def get_dev(x):
        return int(x.value_counts().get(device, 0) > 0)
    return get_dev


class UserFeatureExtractor:
    """
    Класс для добавления фич

    Добавляет фичи
    - has_{os}: есть ли у человека устройство с такой OS
    - has_app_installed: есть ли мобильное приложение
    - uses_{browser}: есть ли у человека определенный браузер
    - region_count: сколько регионов посещал
    """

    def fit(self):
        pass

    def transform(self, events, features):
        print('Applying UserFeatureExtractor...')
        devices = ['Android', 'Windows', 'Mac', 'iOS', 'iPadOS']
        has_devices = events.groupby('viewer_uid').agg(
            **{f'has_{device.lower()}': ('ua_os', _get_get_dev(device)) for device in devices}
        )

        has_other = events.groupby('viewer_uid').agg(
            has_other = ('ua_os', lambda x: int(sum([x.value_counts().get(d, 0) for d in devices]) == 0))
        )

        has_app_installed = events.groupby('viewer_uid').agg(
            has_app_installed = ('ua_client_type', lambda x: int(x.value_counts().get('mobile app', 0) > 0))
        )

        travel_count = events.groupby('viewer_uid').agg(
            travel_count = ('region', lambda x: int(x.nunique()))
        )
        
        browsers = ['Chrome', 'Safari', 'Yandex', 'Firefox', 'Opera', 'Edge', 'Samsung', 'Atom']
        for browser in browsers:
            has_browser = events.groupby('viewer_uid').agg(
                **{f'uses_{browser.lower()}': ('ua_client_name', lambda x: int(x.str.contains(browser).sum() > 0))}
            )
            features = pd.merge(has_browser, features, on='viewer_uid', how='inner')

        features = pd.merge(has_devices, features, on='viewer_uid', how='inner')
        features = pd.merge(has_other, features, on='viewer_uid', how='inner')
        features = pd.merge(has_app_installed, features, on='viewer_uid', how='inner')
        features = pd.merge(travel_count, features, on='viewer_uid', how='inner')

        return events, features

    def fit_transform(self, events, features):
        self.fit()
        return self.transform(events, features)