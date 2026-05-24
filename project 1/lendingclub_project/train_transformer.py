#!/usr/bin/env python
"""Train FT-Transformer and MLP for LendingClub interest rate prediction."""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
warnings.filterwarnings('ignore')

# ============================================================
# REDEFINE PREPROCESSOR CLASS (required for unpickling)
# ============================================================
class LendingClubPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.top_emp_titles = None
    def fit(self, X, y=None):
        emp_title_clean = X['emp_title'].fillna('Unknown').str.lower().str.strip()
        meaningless = ['', 'n/a', 'na', 'none', 'null', 'self', 'employee', 'manager', 'missing']
        emp_title_clean = emp_title_clean.replace(meaningless, 'Unknown')
        self.top_emp_titles = emp_title_clean.value_counts().head(40).index.tolist()
        return self
    def transform(self, X):
        X = X.copy()
        X['fico_midpoint'] = (X['fico_range_low'] + X['fico_range_high']) / 2
        X['fico_category'] = pd.cut(X['fico_midpoint'], bins=[0, 580, 670, 740, 800, 850],
                                     labels=['Poor', 'Fair', 'Good', 'VeryGood', 'Exceptional']).astype(str)
        X['revol_util_ratio'] = X['revol_util'] / 100.0
        X['has_revol_bal'] = (X['revol_bal'] > 0).astype(int)
        X['loan_to_income'] = X['loan_amnt'] / (X['annual_inc'] + 1)
        X['income_log'] = np.log1p(X['annual_inc'])
        X['income_category'] = pd.cut(X['annual_inc'], bins=[0, 40000, 70000, 100000, 150000, float('inf')],
                                       labels=['Low', 'Medium', 'High', 'VeryHigh', 'UltraHigh']).astype(str)
        X['total_bal_log'] = np.log1p(X['tot_cur_bal'].fillna(0))
        X['bal_per_account'] = X['tot_cur_bal'].fillna(0) / (X['total_acc'] + 1)
        X['has_mortgage'] = (X['mort_acc'] > 0).astype(int)
        X['inq_total'] = X['inq_fi'] + X['inq_last_12m']
        X['high_inquiry'] = (X['inq_last_12m'] >= 4).astype(int)
        X['has_delinquency'] = (X['delinq_2yrs'] > 0).astype(int)
        X['has_public_record'] = (X['pub_rec'] > 0).astype(int)
        X['has_bankruptcy'] = (X['pub_rec_bankruptcies'] > 0).astype(int)
        X['has_collection'] = (X['collections_12_mths_ex_med'] > 0).astype(int)
        for col in ['mths_since_last_record', 'mths_since_rcnt_il', 
                     'mths_since_recent_bc', 'mths_since_recent_inq']:
            X[f'{col}_missing'] = X[col].isnull().astype(int)
            X[col] = X[col].fillna(999)
        X['mo_sin_old_il_acct'] = X['mo_sin_old_il_acct'].fillna(0)
        X['credit_depth'] = X['mo_sin_old_rev_tl_op'] + X['mo_sin_old_il_acct']
        X['open_to_total_ratio'] = X['open_acc'] / (X['total_acc'] + 1)
        emp_length_map = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
            '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
            '8 years': 8, '9 years': 9, '10+ years': 10}
        X['emp_length_years'] = X['emp_length'].map(emp_length_map).fillna(0)
        X['emp_length_missing'] = X['emp_length'].isnull().astype(int)
        X['term_months'] = X['term'].map({' 36 months': 36, ' 60 months': 60}).fillna(36)
        emp_title_clean = X['emp_title'].fillna('Unknown').str.lower().str.strip()
        meaningless = ['', 'n/a', 'na', 'none', 'null', 'self', 'employee', 'manager', 'missing']
        emp_title_clean = emp_title_clean.replace(meaningless, 'Unknown')
        X['emp_title_grouped'] = emp_title_clean.apply(lambda x: x if x in self.top_emp_titles else 'Other')
        X['emp_title_missing'] = (emp_title_clean == 'Unknown').astype(int)
        X['zip_prefix'] = X['zip_code'].str[:2]
        drop_cols = ['emp_title', 'emp_length', 'fico_range_low', 'fico_range_high', 'zip_code', 'term', 'revol_util']
        return X.drop(columns=[c for c in drop_cols if c in X.columns])

# ============================================================
# DATA LOADING
# ============================================================
train_df = pd.read_csv('/mnt/agents/upload/LC_train.csv')
test_df = pd.read_csv('/mnt/agents/upload/LC_test.csv')
y = train_df['int_rate'].copy()
test_ids = test_df['ID'].copy()

train_features = train_df.drop(columns=['loan_status', 'title', 'int_rate']).copy()
test_features = test_df.drop(columns=['loan_status', 'title', 'ID']).copy()

n = len(train_features)
train_end, val_end = int(n * 0.70), int(n * 0.85)
X_train = train_features.iloc[:train_end].reset_index(drop=True).copy()
X_val = train_features.iloc[train_end:val_end].reset_index(drop=True).copy()
X_test = train_features.iloc[val_end:].reset_index(drop=True).copy()
y_train = y.iloc[:train_end].reset_index(drop=True).copy()
y_val = y.iloc[train_end:val_end].reset_index(drop=True).copy()
y_test = y.iloc[val_end:].reset_index(drop=True).copy()

# Load preprocessor
with open('/mnt/agents/output/lendingclub_project/models/preprocessor_fe.pkl', 'rb') as f:
    preprocessor_fe = pickle.load(f)

# ============================================================
# PREPROCESSING
# ============================================================
X_train_fe = preprocessor_fe.transform(X_train)
X_val_fe = preprocessor_fe.transform(X_val)
X_test_fe = preprocessor_fe.transform(X_test)
test_fe = preprocessor_fe.transform(test_features)

def get_column_types(df):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    binary = [c for c in numeric if set(df[c].dropna().unique()).issubset({0, 1})]
    continuous = [c for c in numeric if c not in binary]
    return continuous, binary, categorical

cont_cols, bin_cols, cat_cols = get_column_types(X_train_fe)

preprocess_pipeline = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), cont_cols),
    ('bin', SimpleImputer(strategy='most_frequent'), bin_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                      ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))]), cat_cols)
], remainder='drop', verbose_feature_names_out=False)

X_train_processed = preprocess_pipeline.fit_transform(X_train_fe)
X_val_processed = preprocess_pipeline.transform(X_val_fe)
X_test_processed = preprocess_pipeline.transform(X_test_fe)
test_processed = preprocess_pipeline.transform(test_fe)

NUM_FEATURES = X_train_processed.shape[1]
print(f"Features: {NUM_FEATURES}")

# ============================================================
# TORCH MODELS
# ============================================================
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class FTTransformer(nn.Module):
    def __init__(self, num_features, d_token=32, n_blocks=2, n_heads=4, d_hidden=64, dropout=0.15):
        super().__init__()
        self.d_token = d_token
        self.feature_embeddings = nn.ModuleList([nn.Linear(1, d_token) for _ in range(num_features)])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_hidden, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token), nn.Linear(d_token, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, 1))
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
    def forward(self, x):
        batch_size = x.shape[0]
        tokens = [embed(x[:, i:i+1]) for i, embed in enumerate(self.feature_embeddings)]
        x_tokens = torch.stack(tokens, dim=1)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x_tokens = torch.cat([cls, x_tokens], dim=1)
        x_encoded = self.transformer(x_tokens)
        return self.head(x_encoded[:, 0, :]).squeeze(-1)

class TabMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ============================================================
# DATA LOADERS
# ============================================================
X_train_t = torch.FloatTensor(np.array(X_train_processed, dtype=np.float32))
y_train_t = torch.FloatTensor(np.array(y_train.values, dtype=np.float32)).unsqueeze(1)
X_val_t = torch.FloatTensor(np.array(X_val_processed, dtype=np.float32))
y_val_t = torch.FloatTensor(np.array(y_val.values, dtype=np.float32)).unsqueeze(1)
X_test_t = torch.FloatTensor(np.array(X_test_processed, dtype=np.float32))
y_test_t = torch.FloatTensor(np.array(y_test.values, dtype=np.float32)).unsqueeze(1)
test_t = torch.FloatTensor(np.array(test_processed, dtype=np.float32))

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=512, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=512, shuffle=False)

# ============================================================
# TRAINING
# ============================================================
def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3, weight_decay=1e-4, patience=8):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    criterion = nn.MSELoss()
    best_val_rmse = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                val_preds.extend(model(xb).numpy())
                val_targets.extend(yb.squeeze().numpy())
        
        val_rmse = np.sqrt(np.mean((np.array(val_targets) - np.array(val_preds))**2))
        scheduler.step(val_rmse)
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}: val_rmse={val_rmse:.4f}, best={best_val_rmse:.4f}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    return model, best_val_rmse

def evaluate_torch(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.extend(model(xb).numpy())
            targets.extend(yb.squeeze().numpy())
    p, t = np.array(preds), np.array(targets)
    return {
        'RMSE': np.sqrt(np.mean((t-p)**2)),
        'MAE': np.mean(np.abs(t-p)),
        'MAPE': np.mean(np.abs((t-p)/t))*100,
        'R2': 1 - np.sum((t-p)**2)/np.sum((t-t.mean())**2),
        'predictions': p
    }

# ============================================================
# TRAIN MODELS
# ============================================================
print("\n" + "="*60)
print("FT-TRANSFORMER")
print("="*60)
ft_model = FTTransformer(num_features=NUM_FEATURES, d_token=32, n_blocks=2, n_heads=4, d_hidden=64, dropout=0.15)
print(f"Parameters: {sum(p.numel() for p in ft_model.parameters()):,}")
ft_model, ft_val_rmse = train_model(ft_model, train_loader, val_loader, epochs=30, lr=5e-3, weight_decay=1e-4, patience=8)
ft_results = evaluate_torch(ft_model, test_loader)
print(f"TEST: RMSE={ft_results['RMSE']:.4f}, MAE={ft_results['MAE']:.4f}, MAPE={ft_results['MAPE']:.2f}%, R2={ft_results['R2']:.4f}")

print("\n" + "="*60)
print("TAB-MLP")
print("="*60)
mlp_model = TabMLP(input_dim=NUM_FEATURES, hidden_dims=[256, 128, 64], dropout=0.2)
print(f"Parameters: {sum(p.numel() for p in mlp_model.parameters()):,}")
mlp_model, mlp_val_rmse = train_model(mlp_model, train_loader, val_loader, epochs=30, lr=1e-3, weight_decay=1e-4, patience=8)
mlp_results = evaluate_torch(mlp_model, test_loader)
print(f"TEST: RMSE={mlp_results['RMSE']:.4f}, MAE={mlp_results['MAE']:.4f}, MAPE={mlp_results['MAPE']:.2f}%, R2={mlp_results['R2']:.4f}")

# ============================================================
# SAVE RESULTS
# ============================================================
results = {
    'ft_transformer': ft_results,
    'tab_mlp': mlp_results,
    'ft_val_rmse': ft_val_rmse,
    'mlp_val_rmse': mlp_val_rmse
}
pickle.dump(results, open('/mnt/agents/output/lendingclub_project/models/transformer_results.pkl', 'wb'))
torch.save(ft_model.state_dict(), '/mnt/agents/output/lendingclub_project/models/model_ft_transformer.pt')
torch.save(mlp_model.state_dict(), '/mnt/agents/output/lendingclub_project/models/model_tab_mlp.pt')

# Also generate predictions
test_loader_final = DataLoader(test_t, batch_size=512, shuffle=False)
ft_model.eval()
mlp_model.eval()
ft_preds, mlp_preds = [], []
with torch.no_grad():
    for xb in test_loader_final:
        ft_preds.extend(ft_model(xb).numpy())
        mlp_preds.extend(mlp_model(xb).numpy())

np.save('/mnt/agents/output/lendingclub_project/models/ft_test_preds.npy', np.array(ft_preds))
np.save('/mnt/agents/output/lendingclub_project/models/mlp_test_preds.npy', np.array(mlp_preds))
print("\nModels, results, and test predictions saved!")
