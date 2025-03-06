import json
import os
import gzip
import yaml

import pandas as pd
from sklearn.model_selection import train_test_split

def load_jsonl_gz(file_path: str) -> pd.DataFrame:
    data = []
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def main():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)['process_data']
    
    sample = params['sample']
    test_size = params['test_size']
    output_dir = 'data/processed'
    

    os.makedirs(output_dir, exist_ok=True)
    reviews_df = load_jsonl_gz('data/raw/Amazon_Fashion.jsonl.gz')
    
    if sample and sample < len(reviews_df):
        reviews_df = reviews_df.sample(n=sample, random_state=params['random_state'])

    train_df, test_df = train_test_split(reviews_df, test_size=test_size, random_state=params['random_state'])
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

if __name__ == '__main__':
    main()
