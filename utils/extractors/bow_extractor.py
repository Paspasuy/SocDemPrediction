from pathlib import Path

import pandas as pd
import ahocorasick
from utils.text_processing import preprocess_text
from tqdm import tqdm

path = Path(__file__).parent.absolute()

WORDS = ['выпуск', 'сезон', 'серия', 'новый', 'битва', 'экстрасенс', 'часть', 'сериал', 'фильм',
         'женский', 'мужской', 'финал', 'история', 'любовь', 'россия', 'обзор', 'хороший', 'невеста',
         'экстра', 'выживать', 'сокровище', 'император', 'ребенок', 'дубай', 'самый', 'возвращение',
         'макияж', 'титан', 'год', 'шоу', 'июнь', 'русский', 'который', 'день', 'мир', 'дом', 'смотреть',
         'весь', 'это', 'vs', 'путешествие', 'свой', 'игра', 'первый', 'бесплатно', 'человек', 'сон',
         'город', 'семья', 'самойлов', 'юля', 'жизнь', 'мастерская', 'топ', 'лицо', 'воля', 'образ',
         'ханночка', 'язык', 'уход', 'мультфильм', 'страшный', 'аниме', 'отчаянный', 'домохозяйка',
         'звезда', 'последний', 'сильный', 'вечерний', 'пацан', 'арт', 'флюид', 'тренд', 'туториал',
         'почему', 'простой', 'устраивать', 'фоллаут', 'становиться', 'однажды', 'богиня', 'второй',
         'черный', 'дмитрий', 'большой', 'озвучка', 'делать', 'папа', 'художник', 'косметика', 'манго',
         'минута', 'рассказывать', 'сша', 'свадьба', 'se', 'секрет', 'автокресло', 'золотой', 'нато',
         'концерт', 'выбирать', 'танк', 'готовить', 'распаковка', 'вести', 'матч', 'остров', 'вещь',
         'класс', 'москва', 'друг', 'лето', 'монтянин', 'вопрос', 'пучок', 'ч', 'скиталец', 'поделка',
         'час', 'война', 'дело', 'играть', 'эфир', 'челлендж', 'сверхъестественный', 'глаз', 'стиль'
         ]

class BagOfWordsExtractor:
    """
    Класс для добавления фич

    Добавляет в фичи количество видео содеражищих фиксированные слова в названии
    """
    def __init__(self):
        self.user_embed = None
        self.ac = ahocorasick.Automaton()

    def fit(self, events, features):
        for word in WORDS:
            self.ac.add_word(word, word)
        self.ac.make_automaton()

    def transform(self, events, features):
        print("Applying BagOfWords extractor...")
        
        videos = events.groupby('rutube_video_id')['title'].first()
        videos['preprocessed_title'] = videos.apply(preprocess_text)
        
        prepocessed_titles = videos['preprocessed_title']
        prepocessed_titles.name = 'preprocessed_title'
        
        events = events.merge(prepocessed_titles, on='rutube_video_id', how='left')
        
        events[[word + '_count' for word in WORDS]] = 0
        for i in range(len(events)):
            words_found = set()
            for _, word in self.ac.iter(events['preprocessed_title'].iloc[i]):
                if word not in words_found:
                    events.at[i, word + '_count'] += 1
                    words_found.add(word)
                    
        features = features.merge(events.groupby('viewer_uid')[[word + '_count' for word in WORDS]].sum(),
                                  on='viewer_uid', how='left')
        
        return events, features
    
        

    def fit_transform(self, events, features):
        self.fit(events, features)
        return self.transform(events, features)