import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


LEAKY_FEATURES = ['poutcome', 'pdays', 'duration']


def drop_leaky_features(X_train, X_test):
    present = [f for f in LEAKY_FEATURES if f in X_train.columns]
    if present:
        X_train = X_train.drop(columns=present)
        X_test = X_test.drop(columns=present)
    return X_train, X_test


def engineer_ebm_features(X_train, X_test):
    features_added = []

    # -----------------------------
    # Cyclic temporal encodings
    # -----------------------------
    if 'month' in X_train.columns:
        for df in (X_train, X_test):
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        features_added += ['month_sin', 'month_cos']

    if 'day_of_week' in X_train.columns:
        for df in (X_train, X_test):
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        features_added += ['dow_sin', 'dow_cos']

    # -----------------------------
    # Saturation log transforms
    # -----------------------------
    if 'campaign' in X_train.columns:
        for df in (X_train, X_test):
            df['log_campaign'] = np.log1p(df['campaign'])
        features_added.append('log_campaign')

    if 'previous' in X_train.columns:
        for df in (X_train, X_test):
            df['log_previous'] = np.log1p(df['previous'])
        features_added.append('log_previous')

    # -----------------------------
    # Macro economic composite
    # -----------------------------
    econ_features = [
        'nr.employed', 'euribor3m',
        'emp.var.rate', 'cons.price.idx', 'cons.conf.idx'
    ]
    econ_present = [f for f in econ_features if f in X_train.columns]

    if len(econ_present) >= 3:
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(X_train[econ_present])
        test_scaled = scaler.transform(X_test[econ_present])

        sign_map = {
            'nr.employed': -1,
            'euribor3m': -1,
            'emp.var.rate': -1,
            'cons.price.idx': -1,
            'cons.conf.idx': 1,
        }
        signs = np.array([sign_map.get(f, 1) for f in econ_present])

        X_train['macro_index'] = (train_scaled * signs).mean(axis=1)
        X_test['macro_index'] = (test_scaled * signs).mean(axis=1)
        features_added.append('macro_index')

    # -----------------------------
    # Age polynomial
    # -----------------------------
    if 'age' in X_train.columns:
        mean, std = X_train['age'].mean(), X_train['age'].std()
        for df in (X_train, X_test):
            df['age_centered'] = (df['age'] - mean) / std
            df['age_squared'] = df['age_centered'] ** 2
        features_added += ['age_centered', 'age_squared']

    # -----------------------------
    # Binary indicators
    # -----------------------------
    if 'campaign' in X_train.columns:
        for df in (X_train, X_test):
            df['campaign_fatigue'] = (df['campaign'] > 5).astype('int8')
        features_added.append('campaign_fatigue')

    if 'previous' in X_train.columns:
        for df in (X_train, X_test):
            df['has_previous'] = (df['previous'] > 0).astype('int8')
        features_added.append('has_previous')

    return X_train, X_test, features_added


def prune_redundant_features(X_train, X_test):
    drop_candidates = [
        'day', 'default',
        'age', 'month', 'day_of_week',
        'campaign', 'previous',
        'nr.employed', 'euribor3m',
        'emp.var.rate', 'cons.price.idx', 'cons.conf.idx'
    ]

    present = [f for f in drop_candidates if f in X_train.columns]
    if present:
        X_train = X_train.drop(columns=present)
        X_test = X_test.drop(columns=present)

    return X_train, X_test
