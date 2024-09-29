import os
import torch
from torch import Tensor
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from sklearn.decomposition import PCA


def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    

def reduce_dimensionality(data, target_dimension):
    """
    Уменьшает размерность данных с d до target_dimension с помощью PCA.

    Args:
        data (numpy.ndarray): Исходные данные размерности (n_samples, d).
        target_dimension (int): Желаемая размерность выходных данных.

    Returns:
        numpy.ndarray: Данные размерности (n_samples, target_dimension).
    """
    # Инициализируем PCA с целевой размерностью
    pca = PCA(n_components=target_dimension)
    
    # Применяем PCA к данным
    reduced_data = pca.fit_transform(data)
    
    return reduced_data


def download_embeddings(video, device_ids: list, embeds_size=32, model_hard=False) -> np.array:
    """
    Функция для загрузки эмбеддингов. Есть тяжёлая 1.5B модель, есть лёгкая 560M.
    Проходимся по текстам и достаём эмбеддинги из них моделькой, а затем с помощью
    k_means сжимаем эмбеддинги до размерности embeds_size
    """

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    video = video.copy()

    # Для инструкционных эмбеддингов
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'
    
    model_name = 'intfloat/multilingual-e5-large-instruct'
    if model_hard:
        model_name = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'
    
    if len(device_ids) == 0:
        print('GPU не выбрано!!! Инференс может занимать очень много времени!')
        model_name = 'intfloat/multilingual-e5-large-instruct'
    
    cache_dir = f"./model_cache/{model_name}"

    tokenizer = AutoTokenizer.from_pretrained(cache_dir)
    model = AutoModel.from_pretrained(cache_dir)

    if len(device_ids) > 0:
        model = nn.DataParallel(model, device_ids=device_ids)
        n_gpu = len(device_ids)

        model.to(f"cuda:{device_ids[0]}")
        print('model.device_ids:', model.device_ids)

    max_length = 128  # Максимальная длина токенов в тексте

    def tokenize(texts):
        tensors = tokenizer(texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        return {k : v for k, v in tensors.items()}

    def embed(texts):
        with torch.no_grad():
            t = tokenize(texts)
            last_state = model(**t).last_hidden_state
            return last_token_pool(last_state, t["attention_mask"])

    def embed_batched(texts, bs=256):
        """
        Считает эмбеддинги с большим батчсайзом
        """
        n = len(texts)
        res = []
        for i in trange(0, n, bs):
            res.append(embed(texts[i:i+bs]))
        return list(torch.cat(res).cpu().numpy().squeeze())

    if model_hard:
        video['text'] = video.apply(lambda x: 'Название видео: "' + x['title'] + '"; Категория видео: ' + x['category'], axis=1)
        video['query'] = video['text'].apply(lambda x: get_detailed_instruct("Определи средний возраст целевой аудитории данного видео", x))
        embeds_large = embed_batched(list(video['query']))
        embeds_small = reduce_dimensionality(embeds_large, embeds_size)
        return embeds_small
    
    video['text'] = video.apply(lambda x: x['title'] + '; Категория: ' + x['category'], axis=1)
    embeds = embed_batched(list(video['text']))
    embeds_small = reduce_dimensionality(embeds, embeds_size)
    return embeds_small
