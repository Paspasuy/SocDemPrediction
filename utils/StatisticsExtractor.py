import pandas as pd 
import numpy as np

class SimpleStatisticsExtractor:
    """
    Извлечение простых сатистик из данных
    
    total_views - общее количество просмотров юзера
    
    total_watchtime_{avg, max, min, std, sum}
    percent_watched_{avg, max, min, std}
    duration_{avg, max, min, std}
    
    hour_x - доля активности в часе x
    day_x - доля активности в дне x
    
    category_x_viewtime - доля просмотра категории x по врменени
    category_x_views - количество просмотров категории x по кликам
    
    video_popularity_{avg, max, min, std} - популярность видео просмотренных юзером
    """

    def fit(self):
        pass

    def transform(self, events: pd.DataFrame, features: pd.DataFrame):
        features = features.copy()
        events = events.copy()
        
        total_views = events.groupby('viewer_uid').size().reset_index(name='total_views')
        features = features.merge(total_views, on='viewer_uid', how='left')
        
        total_watchtime = events.groupby('viewer_uid')['total_watchtime'].agg(['mean', 'max', 'min', 'std', 'sum']).reset_index()
        features = features.merge(total_watchtime, on='viewer_uid', how='left')
        
        events['percent_watched'] = events['total_watchtime'] / events['duration'] * 1000
        percent_watched = events.groupby('viewer_uid')['percent_watched'].agg(['mean', 'max', 'min', 'std']).reset_index()
        features = features.merge(percent_watched, on='viewer_uid', how='left')
        
        duration = events.groupby('viewer_uid')['duration'].agg(['mean', 'max', 'min', 'std']).reset_index()
        features = features.merge(duration, on='viewer_uid', how='left')
        
        events['hour'] = events['local_time'].dt.hour
        events['day'] = events['local_time'].dt.weekday
        for i in range(24):
            hour_i = events[events['hour'] == i].groupby('viewer_uid')['total_watchtime'].transform('sum')
            features[f'hour_{i}'] = hour_i / events.groupby('viewer_uid')['total_watchtime'].transform('sum')
        for i in range(7):
            day_i = events[events['day'] == i].groupby('viewer_uid')['total_watchtime'].transform('sum')
            features[f'day_{i}'] = day_i / events.groupby('viewer_uid')['total_watchtime'].transform('sum')
        
        category_viewtime = events.groupby(['viewer_uid', 'category'])['total_watchtime'].transform('sum')
        category_views = events.groupby(['viewer_uid', 'category']).size()
        category_viewtime = category_viewtime / events.groupby('viewer_uid')['total_watchtime'].transform('sum')
        category_views = category_views / events.groupby('viewer_uid').size()
        for category in events['category'].unique():
            features[f'category_{category}_viewtime'] = category_viewtime[events['category'] == category]
            features[f'category_{category}_views'] = category_views[events['category'] == category]
            
        return events, features
        

    def fit_transform(self, events: pd.DataFrame, features: pd.DataFrame):
        self.fit()
        return self.transform(events, features)
    
    
if __name__ == '__main__':
    events = pd.read_csv('data/train_events.csv')
    video = pd.read_csv('data/video_info_v2.csv')
    events = events.merge(video, on='rutube_video_id', how='left')
    targets = pd.read_csv('data/train_targets.csv')
    
    extractor = SimpleStatisticsExtractor()
    events, features = extractor.fit_transform(events, targets)
    print(features.head())