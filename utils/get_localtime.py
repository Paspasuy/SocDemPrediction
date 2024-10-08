from pathlib import Path

import pandas as pd
import pytz

path = Path(__file__).parent.absolute()

_reg_to_tz = pd.read_csv(path / 'regions_result.csv')
_timezones = {tz: pytz.timezone(tz) for tz in _reg_to_tz['timezone']}

# Функция для преобразования UTC времени в местное с учетом таймзоны
def _convert_to_local_time(row):
    return row['event_timestamp'].astimezone(_timezones[row['timezone']])


def add_tz_and_localtime_column(df):
    region_to_timezone_map = dict(zip(_reg_to_tz['region'], _reg_to_tz['timezone']))
    df['timezone'] = df['region'].apply(lambda region: region_to_timezone_map[region])
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])

    df['local_time'] = df.apply(_convert_to_local_time, axis=1)
    df['utc_delta'] = df['local_time'].apply(lambda x: x.utcoffset().seconds // 3600)
    df['local_time'] = df['local_time'].apply(lambda x: x.replace(tzinfo=None))
    df['local_time'] = pd.to_datetime(df["local_time"], utc=True)

    return df
