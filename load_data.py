import os
import requests
import pathlib

BASE_DIR = pathlib.Path(__file__).parent

GET_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz"


def load_url(url: str) -> bytes:
    content = requests.get(url).content
    return content


def load_data():
    save_dir = os.path.join(BASE_DIR, 'data/raw/')
    save_path = os.path.join(save_dir, 'Amazon_Fashion.jsonl.gz')

    if not os.path.exists(save_path):
        dataset = load_url(GET_URL)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(save_path, 'wb') as f:
            f.write(dataset)


if __name__ == '__main__':
    load_data()
