# Скрипт, который использовался для создания маппинга reginos_result.csv: Название региона -> таймзона.

from timezonefinder import TimezoneFinder
from geopy.geocoders import Nominatim
import time
import pandas as pd

path = "./"
regions_df = pd.read_csv(path + 'regions.csv')

# Инициализация необходимых инструментов
tf = TimezoneFinder()
geolocator = Nominatim(user_agent="timezone_extractor")

# Функция для получения часового пояса по названию региона
def get_timezone(region_name):
    print(region_name)
    try:
        # Получаем координаты региона
        location = geolocator.geocode(region_name + ", Russia")
        if location:
            # Получаем часовой пояс по координатам
            timezone = tf.timezone_at(lat=location.latitude, lng=location.longitude)
            print(location, timezone)
            return timezone
        else:
            return "Не найдено"
    except Exception as e:
        return str(e)

# Применяем функцию ко всем регионам
regions_df['timezone'] = regions_df['region'].apply(lambda x: get_timezone(x))

# Сохраним результат в таблице и выведем несколько первых строк
regions_df.head()

regions_df.to_csv("regions_result.csv")
