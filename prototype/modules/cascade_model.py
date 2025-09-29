
# =============================================================================
# IMPROVED: /content/churn_pipeline/modules/cascade_model.py
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin

# =========================
# RNN Wrapper
# =========================
class SimpleRNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Simple RNN classifier using PyTorch, wrapped for sklearn compatibility.
    Treats each feature as a time step in a sequence for temporal pattern learning.
    """
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.3, learning_rate=0.001, epochs=100, batch_size=32, random_state=42):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None
        self.input_size = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _set_seed(self):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

    class RNNModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout):
            super().__init__()
            self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_size,
                               num_layers=num_layers, dropout=dropout, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            x = x.unsqueeze(2)  # (batch, seq_len, 1)
            rnn_out, _ = self.rnn(x)
            out = self.dropout(rnn_out[:, -1, :])
            out = self.fc(out)
            out = self.sigmoid(out)
            return out

    def fit(self, X, y):
        self._set_seed()
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y).reshape(-1, 1) if not isinstance(y, np.ndarray) else y.reshape(-1, 1)
        self.input_size = X.shape[1]

        self.model = self.RNNModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(0, len(X_tensor), self.batch_size):
                batch_X = X_tensor[i:i+self.batch_size]
                batch_y = y_tensor[i:i+self.batch_size]

                optimizer.zero_grad()
                loss = criterion(self.model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 20 == 0:
                print(f"RNN Epoch {epoch}/{self.epochs}, Loss: {total_loss/len(X_tensor):.4f}")
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not fitted yet")
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            proba = self.model(X_tensor).cpu().numpy()
        return np.column_stack([1 - proba, proba])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


# =========================
# Cascade Model
# =========================
class CascadeModel:
    """
    Cascade ensemble model with balanced sampling strategy:
    Stage 1: Random Forest (robust baseline)
    Stage 2: Neural Network (complex patterns)  
    Stage 3: RNN (sequential pattern recognition)
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.stage1_model = None
        self.stage2_model = None
        self.stage3_model = None
        self.sampling_pipeline = None

    def _create_balanced_sampling_pipeline(self):
        smote = BorderlineSMOTE(random_state=self.random_state, sampling_strategy=0.6)
        undersample = RandomUnderSampler(random_state=self.random_state, sampling_strategy=0.75)
        self.sampling_pipeline = ImbPipeline([
            ('smote', smote),
            ('under', undersample)
        ])
        return self.sampling_pipeline

    def train_cascade_pipeline(self, X_train_preprocessed, y_train, X_test_preprocessed, y_test):
        print(f"\n=== TRAINING CASCADE PIPELINE ===")
        print(f"Original class distribution:")
        print(f"  Class 0: {(y_train == 0).sum()}, Class 1: {(y_train == 1).sum()}")

        # Ensure inputs are numpy arrays
        X_train_preprocessed = np.array(X_train_preprocessed)
        X_test_preprocessed = np.array(X_test_preprocessed)
        y_train = np.array(y_train)

        # Balanced sampling
        try:
            pipeline = self._create_balanced_sampling_pipeline()
            X_train_bal, y_train_bal = pipeline.fit_resample(X_train_preprocessed, y_train)
            print(f"After balanced sampling:")
            print(f"  Class 0: {(y_train_bal == 0).sum()}, Class 1: {(y_train_bal == 1).sum()}")
            print(f"  Total samples: {len(y_train_bal)} (was {len(y_train)})")
        except Exception as e:
            print(f"Balanced sampling failed: {e}, using original data")
            X_train_bal, y_train_bal = X_train_preprocessed, y_train

        # Stage 1: Random Forest
        print("\nTraining Stage 1: Random Forest...")
        self.stage1_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced_subsample',
            random_state=self.random_state,
            n_jobs=-1
        )
        self.stage1_model.fit(X_train_bal, y_train_bal)

        # Stage 2: Neural Network
        print("Training Stage 2: MLP Neural Network...")
        self.stage2_model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=self.random_state
        )
        self.stage2_model.fit(X_train_bal, y_train_bal)

        # Stage 3: RNN
        print("Training Stage 3: RNN...")
        self.stage3_model = SimpleRNNClassifier(
            hidden_size=64,
            num_layers=2,
            dropout=0.3,
            learning_rate=0.001,
            epochs=100,
            batch_size=64,
            random_state=self.random_state
        )
        self.stage3_model.fit(X_train_bal, y_train_bal)

        # Predictions
        print("\nMaking predictions on test set...")
        y_proba1 = self.stage1_model.predict_proba(X_test_preprocessed)[:, 1]
        y_proba2 = self.stage2_model.predict_proba(X_test_preprocessed)[:, 1]
        y_proba3 = self.stage3_model.predict_proba(X_test_preprocessed)[:, 1]

        # Weighted ensemble
        y_proba_final = y_proba1*0.3 + y_proba2*0.3 + y_proba3*0.4
        threshold = 0.45
        y_pred = (y_proba_final > threshold).astype(int)

        print(f"Using threshold: {threshold}")
        print(f"Predicted churners: {y_pred.sum()} out of {len(y_pred)} customers")

        return y_test, y_pred, y_proba_final

    def get_feature_importance(self):
        if self.stage1_model is not None:
            return self.stage1_model.feature_importances_
        return None
