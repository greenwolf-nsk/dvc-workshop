### Инструкция по запуску

Рекомендуется использовать Python версии 3.11+ и виртуальное окружение.

0. Скачать файл [Amazon_Fashion.jsonl.gz](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz) и положить в `data/raw`

1. Установить зависимости:
```bash
pip install -r requirements.txt
```

2. Инициализировать DVC:
```bash
dvc init -f
```

3. Запустить пайплайн:
```bash
dvc repro
```

4. Просмотреть метрики:
```bash
dvc metrics show
```