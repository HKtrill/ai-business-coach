# imports + small utilities (cell 1)
import os
import json
import random
import hashlib
import joblib
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, precision_recall_curve, auc,
                             average_precision_score)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Utility: set all relevant RNG seeds for reproducibility
def set_seed(seed):
    """Set seed for python random, numpy, and environment hash seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Utility: ensure all artifact directories exist
def ensure_dirs(base_dir="artifacts"):
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "preprocessors"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
    os.makedirs("data_splits", exist_ok=True)

ensure_dirs()