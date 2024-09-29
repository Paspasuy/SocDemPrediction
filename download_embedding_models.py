from transformers import AutoTokenizer, AutoModel
import os

model_names = [
    'intfloat/multilingual-e5-large-instruct',
    'Alibaba-NLP/gte-Qwen2-1.5B-instruct'
]
cache_dir = "./model_cache"

for model_name in model_names:
    # Создаем директорию для сохранения модели, если не существует
    os.makedirs(cache_dir + f'/{model_name}', exist_ok=True)

    # Загрузка и сохранение токенизатора
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.save_pretrained(cache_dir)

    # Загрузка и сохранение модели
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    model.save_pretrained(cache_dir)
