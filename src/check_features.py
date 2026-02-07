import os
import sys
import joblib

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
features = joblib.load(os.path.join(DATA_DIR, "feature_columns.joblib"))

print('Number of features:', len(features))
print('\nFeature list:')
print('\n'.join(features))
print('\n\nChecking for data leakage...')
leaky = [f for f in features if 'future' in f.lower() or 'target' in f.lower()]
print('Leaky features found:', leaky if leaky else 'None - GOOD!')
