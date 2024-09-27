import pandas as pd
import pytz
from datetime import datetime


_reg_to_tz = pd.read_csv("regions_result.csv")
_timezones = {tz: pytz.timezone(tz) for tz in _reg_to_tz['timezone']}

# Функция для преобразования UTC времени в местное с учетом таймзоны
def _convert_to_local_time(row):
    return row['event_timestamp'].astimezone(_timezones[row['timezone']])


def add_tz_and_localtime_column(df):
    region_to_timezone_map = dict(zip(_reg_to_tz['region'], _reg_to_tz['timezone']))
    df['timezone'] = df['region'].apply(lambda region: region_to_timezone_map[region])
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])

    df['local_time'] = df.apply(_convert_to_local_time, axis=1)
    return df
