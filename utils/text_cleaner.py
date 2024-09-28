import torch
import os
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import pandas as pd
import seaborn as sns
import numpy as np
from tqdm import tqdm, trange

import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pymystem3
tqdm.pandas()

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
russian_stopwords = stopwords.words('russian')
stemmer = pymystem3.Mystem()


def preprocess_text(text):
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление ненужных символов
    text = re.sub(r'[^a-zа-яё0-9\s]', '', text)
    text = ''.join(stemmer.lemmatize(text))
    return text


# merged_data['cleaned_text'] = merged_data['text'].progress_apply(preprocess_text)