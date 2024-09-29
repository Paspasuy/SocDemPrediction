import pandas as pd 
import numpy as np

class SimpleStatisticsExtractor:
    """
    Извлечение простых сатистик из данных
    
    total_views - общее количество просмотров юзера
    
    total_watchtime_{avg, max, min, std, sum}
    percent_watched_{avg, max, min, std}
    duration_{avg, max, min, std}
    
    percent_skipped - доля пропущенных видео (где percent_watched < 0.1)
    percent_completed - доля полностью просмотренных видео (где percent_watched > 0.95)
    percent_rewatched - доля пересмотренных видео (где percent_watched > 1.1)
    
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
        
        print('Applying SimpleStatisticsExtractor...')
                      
        total_views = events.groupby('viewer_uid').size().reset_index(name='total_views')
        features = features.merge(total_views, on='viewer_uid', how='left')
        
        total_watchtime = events.groupby('viewer_uid')['total_watchtime'].agg(['mean', 'max', 'min', 'std', 'sum'])
        total_watchtime.columns = [f'total_watchtime_{col}' for col in total_watchtime.columns]
        features = features.merge(total_watchtime, on='viewer_uid', how='left')
        
        events['percent_watched'] = events['total_watchtime'] / events['duration'] * 1000
        percent_watched = events.groupby('viewer_uid')['percent_watched'].agg(['mean', 'max', 'min', 'std'])
        percent_watched.columns = [f'percent_watched_{col}' for col in percent_watched.columns]
        features = features.merge(percent_watched, on='viewer_uid', how='left')
        
        events['watched'] = events['percent_watched'] > 0.95
        events['skipped'] = events['percent_watched'] < 0.1
        events['rewatched'] = events['percent_watched'] > 1.1
        percent_skipped = events.groupby('viewer_uid')['skipped'].mean()
        percent_skipped.name = 'percent_skipped'
        percent_completed = events.groupby('viewer_uid')['watched'].mean()
        percent_completed.name = 'percent_completed'
        percent_rewatched = events.groupby('viewer_uid')['rewatched'].mean()
        percent_rewatched.name = 'percent_rewatched'
        features = features.merge(percent_skipped, on='viewer_uid', how='left')
        features = features.merge(percent_completed, on='viewer_uid', how='left')
        features = features.merge(percent_rewatched, on='viewer_uid', how='left')
        
        duration = events.groupby('viewer_uid')['duration'].agg(['mean', 'max', 'min', 'std'])
        duration.columns = [f'duration_{col}' for col in duration.columns]
        features = features.merge(duration, on='viewer_uid', how='left')
        
        events['hour'] = events['local_time'].dt.hour
        events['day'] = events['local_time'].dt.weekday
        for i in range(24):
            hour_i = events[events['hour'] == i].groupby('viewer_uid')['total_watchtime'].sum()
            hour_i = hour_i / events.groupby('viewer_uid')['total_watchtime'].sum()
            hour_i.name = f'hour_{i}'
            features = features.merge(hour_i, on='viewer_uid', how='left')
        for i in range(7):
            day_i = events[events['day'] == i].groupby('viewer_uid')['total_watchtime'].sum()
            day_i = day_i / events.groupby('viewer_uid')['total_watchtime'].sum()
            day_i.name = f'day_{i}'
            features = features.merge(day_i, on='viewer_uid', how='left')
                
        category_viewtime = events.groupby(['viewer_uid', 'category'])['total_watchtime'].sum()
        category_views = events.groupby(['viewer_uid', 'category']).size()
        category_viewtime = category_viewtime / events.groupby('viewer_uid')['total_watchtime'].sum()
        category_views = category_views / events.groupby('viewer_uid').size()
        for category in events['category'].unique():
            viewtime = category_viewtime.xs(category, level='category')
            viewtime.name = f'category_{category}_viewtime'
            views = category_views.xs(category, level='category')
            views.name = f'category_{category}_views'
            features = features.merge(viewtime, on='viewer_uid', how='left').fillna(0)
            features = features.merge(views, on='viewer_uid', how='left').fillna(0)
            
            
        return events, features
        

    def fit_transform(self, events: pd.DataFrame, features: pd.DataFrame):
        self.fit()
        return self.transform(events, features)
    
