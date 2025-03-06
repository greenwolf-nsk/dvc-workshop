import os
import json
import yaml
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

def train_model():
    os.makedirs('models', exist_ok=True)
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)['lightgbm']
    
    train_features = sp.load_npz('data/features/train_features.npz')
    test_features = sp.load_npz('data/features/test_features.npz')
    train_target = pd.read_csv('data/features/train_target.csv')['rating']
    test_target = pd.read_csv('data/features/test_target.csv')['rating']
    
    model = LGBMRegressor(
        n_estimators=params.get('n_estimators', 100),
        learning_rate=params.get('learning_rate', 0.01),
        max_depth=params.get('max_depth', 10),
        random_state=params.get('random_state', 42)
    )
    model.fit(train_features, train_target)
    
    train_preds = model.predict(train_features)
    test_preds = model.predict(test_features)
    train_mse = np.sqrt(mean_squared_error(train_target, train_preds))
    test_mse = np.sqrt(mean_squared_error(test_target, test_preds))
    metrics = {
        'train_mse': train_mse,
        'test_mse': test_mse
    }
    
    with open('lightgbm_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    with open('models/lightgbm_model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    train_model()
