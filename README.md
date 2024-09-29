# SocDemPrediction
Решение команды ThreeNearestNeighbours (MIPT, MISIS)

## Общая схема решения

Сначала мы провели визуальный анализ данных [вставить ссылку на ноутбук] что позволило нам поределиться со стратегией решения и выбрать мнгообещающие подходы

Мы решили использовать следующую арихтектуру: в [вставить ссылку] лежат классы FeatureExtractor'ов которые занимаются созданием различных фичей. Далее полученные фичи объединяются в одну таблицу и подаются на вход CatBoost (мы обучаем две модели - одну для предсказания пола и одну для возраста). Модель получается как усреднение нескольких моделей обученных по методу кросс-валидации. Полученный результат: ...

Перед всеми экстракторами так же применяется Imputer из файла imputer.py - там в том числе находится логика преобразования времени в местное.

Далее приводятся описания различных экстракторов которые мы использовали:

- **SimpleStatisticsExtractor** - генерация простых статитстических фичей, таких как средний процент просмотра видео, активность по часам в дне и по дням недели и так далее. Признаки завязанные на процент просомтра видео (минимальный, средний, ...) оказались довольно значимыми
- **UserFeatureExtractor** - фичи основанные на устройствах пользователя, например есть ли у него айфон (значимый признак - [описать как влияет]). Из интересного добавили признак считающий сколько регионов посетил пользователь за всю историю. Предположительно это влияет на возраст [подтвердить]
- **GeoExtractor** - географические признаки, такие как число пользователей в регионе и часовой пояс
- **MainExtractor** - здесь мы добавляем для каждого пользователя эмбеддинг, полученный усреднением всех эмбеддингов названий видео, которые посмотрел пользователь: чем больше пользователь посмотрел видео, тем больший вес имеет видео в сумме эмбеддингов
- **TargetEncodingExtractor** - 
- **BagOfWordsExtractor** - при анализе названий видео мы заметили тенденцию что некоторые сериалы/шоу смотрят в основном женщины, а некоторые - мужчины. Мы не были уверены что наши эмбеддинги могут ухватить это, поэтому нормализовав текст выделили 128 самых частых слов, за исключением стоп-слов вроде предлогов. В эти слова в основном вошли названия сериалов и самых популярных телешоу - именно то чего мы и хотели. Далее в экстракторе с помощью алгоритма Ахо-Корасик считается число вхождений искомых слов в названия видео для пользователя и все это подается в модель. В результате модель учла что, например, про фоллаут смотрят в основном мужчины.
- **ALSEmbeddingExtractor** - посмотрев на данные можно заметить что видео лишь 2% авторов генерируют X% событий. Возникает идея представить авторов и пользователей как вектора с помощью метода AlternatingLeastSquares, что будет работать очень быстро за счет маленького числа важных авторов и, к тому же, не будет использовать target, что позволяет в полной мере использовать все данные. В ходе экспериментов были получены довольно любопытные результаты с которыми можно ознакомиться в ноутбуке als.ipynb. Полученные 64-мерные представления пользователей были переданы в модель. Еще один плюс ALS что помимо классификации пользователей мы получаем готовую модель для рекомендации контента.

В конце мы использовали подбор гиперпараметров CatBoost с помощью Optuna чтобы еще немного улучшить наш рещультат

## Анализ точности в зависимости от длины истории

## Как запустить

## Инсайты полученные при анализе

## Что еще пробовали

Пробовали кучу способов обрабатывать текст, ничего, кроме эмбеддингов, не дало свои плоды: personal/knifeman/video_embeddings.ipynb

## 

