import os
import yaml

import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

def create_features():
    os.makedirs('data/features', exist_ok=True)
    
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)['create_features']
    
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    vectorizer = CountVectorizer(max_features=params['max_features'])
    vectorizer.fit(train_df['text'].fillna(''))
    
    train_features = vectorizer.transform(train_df['text'].fillna(''))
    test_features = vectorizer.transform(test_df['text'].fillna(''))
    
    sp.save_npz('data/features/train_features.npz', train_features)
    sp.save_npz('data/features/test_features.npz', test_features)
    
    train_target = train_df['rating'].astype(float)
    train_target.to_csv('data/features/train_target.csv', index=False, header=True)
    test_target = test_df['rating'].astype(float)
    test_target.to_csv('data/features/test_target.csv', index=False, header=True)
    

if __name__ == '__main__':
    create_features()
