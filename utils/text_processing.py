from tqdm import tqdm
import pandas as pd

import re
import pymystem3
from collections import Counter

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stopwords = stopwords.words('russian') + stopwords.words('english')
stemmer = pymystem3.Mystem()


def preprocess_text(text: str) -> str:
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление ненужных символов
    text = re.sub(r'[^a-zа-яё\s]', '', text)
    text = ''.join(stemmer.lemmatize(text)).split()
    # Удаление стоп-слов
    text = [word for word in text if word not in stopwords]
    return ' '.join(text)


def get_top_words(texts: list, top_n: int = 128) -> list:
    words = []
    for text in tqdm(texts):
        words.extend(preprocess_text(text).split())
    return [word for word, _ in Counter(words).most_common(top_n)]

if __name__ == '__main__':
    all_events = pd.read_csv('data/all_events.csv')
    train_events = pd.read_csv('data/train_events.csv')
    video_info = pd.read_csv('data/video_info_v2.csv')
    
    train_events = train_events.merge(video_info, on='rutube_video_id', how='left')
    all_events = all_events.merge(video_info, on='rutube_video_id', how='left')
    all_events = pd.concat([all_events, train_events])
    
    all_events = all_events.sample(frac=1).reset_index(drop=True)
    sample_titles = all_events.groupby('viewer_uid')['title'].first().reset_index()
    top_words = get_top_words(sample_titles['title'].tolist())
    print(top_words)
    
    