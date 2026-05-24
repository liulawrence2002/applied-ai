#!/usr/bin/env python
"""Train neural network models for LendingClub interest rate prediction."""
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

# Preprocessor class (needed for unpickling)
class LendingClubPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self): self.top_emp_titles = None
    def fit(self, X, y=None):
        emp = X['emp_title'].fillna('Unknown').str.lower().str.strip()
        for m in ['', 'n/a', 'na', 'none', 'null', 'self', 'employee', 'manager', 'missing']:
            emp = emp.replace(m, 'Unknown')
        self.top_emp_titles = emp.value_counts().head(40).index.tolist()
        return self
    def transform(self, X):
        X = X.copy()
        X['fico_midpoint'] = (X['fico_range_low'] + X['fico_range_high']) / 2
        X['fico_category'] = pd.cut(X['fico_midpoint'], bins=[0,580,670,740,800,850],
            labels=['Poor','Fair','Good','VeryGood','Exceptional']).astype(str)
        X['revol_util_ratio'] = X['revol_util'] / 100.0
        X['has_revol_bal'] = (X['revol_bal'] > 0).astype(int)
        X['loan_to_income'] = X['loan_amnt'] / (X['annual_inc'] + 1)
        X['income_log'] = np.log1p(X['annual_inc'])
        X['income_category'] = pd.cut(X['annual_inc'], bins=[0,40000,70000,100000,150000,np.inf],
            labels=['Low','Medium','High','VeryHigh','UltraHigh']).astype(str)
        X['total_bal_log'] = np.log1p(X['tot_cur_bal'].fillna(0))
        X['has_mortgage'] = (X['mort_acc'] > 0).astype(int)
        X['inq_total'] = X['inq_fi'] + X['inq_last_12m']
        X['high_inquiry'] = (X['inq_last_12m'] >= 4).astype(int)
        X['has_delinquency'] = (X['delinq_2yrs'] > 0).astype(int)
        X['has_public_record'] = (X['pub_rec'] > 0).astype(int)
        X['has_bankruptcy'] = (X['pub_rec_bankruptcies'] > 0).astype(int)
        X['has_collection'] = (X['collections_12_mths_ex_med'] > 0).astype(int)
        for c in ['mths_since_last_record','mths_since_rcnt_il','mths_since_recent_bc','mths_since_recent_inq']:
            X[f'{c}_missing'] = X[c].isnull().astype(int)
            X[c] = X[c].fillna(999)
        X['mo_sin_old_il_acct'] = X['mo_sin_old_il_acct'].fillna(0)
        X['credit_depth'] = X['mo_sin_old_rev_tl_op'] + X['mo_sin_old_il_acct']
        X['open_to_total_ratio'] = X['open_acc'] / (X['total_acc'] + 1)
        el = {'< 1 year':0,'1 year':1,'2 years':2,'3 years':3,'4 years':4,'5 years':5,
              '6 years':6,'7 years':7,'8 years':8,'9 years':9,'10+ years':10}
        X['emp_length_years'] = X['emp_length'].map(el).fillna(0)
        X['emp_length_missing'] = X['emp_length'].isnull().astype(int)
        X['term_months'] = X['term'].map({' 36 months':36,' 60 months':60}).fillna(36)
        et = X['emp_title'].fillna('Unknown').str.lower().str.strip()
        for m in ['','n/a','na','none','null','self','employee','manager','missing']:
            et = et.replace(m, 'Unknown')
        X['emp_title_grouped'] = et.apply(lambda x: x if x in self.top_emp_titles else 'Other')
        X['emp_title_missing'] = (et == 'Unknown').astype(int)
        X['zip_prefix'] = X['zip_code'].str[:2]
        dc = ['emp_title','emp_length','fico_range_low','fico_range_high','zip_code','term','revol_util']
        return X.drop(columns=[c for c in dc if c in X.columns])

# Load data
train_df = pd.read_csv('/mnt/agents/upload/LC_train.csv')
test_df = pd.read_csv('/mnt/agents/upload/LC_test.csv')
y = train_df['int_rate'].copy()
test_ids = test_df['ID'].copy()

train_features = train_df.drop(columns=['loan_status','title','int_rate']).copy()
test_features = test_df.drop(columns=['loan_status','title','ID']).copy()

n = len(train_features)
te, ve = int(n*0.70), int(n*0.85)
X_train = train_features.iloc[:te].reset_index(drop=True).copy()
X_val = train_features.iloc[te:ve].reset_index(drop=True).copy()
X_test = train_features.iloc[ve:].reset_index(drop=True).copy()
y_train = y.iloc[:te].reset_index(drop=True).copy()
y_val = y.iloc[te:ve].reset_index(drop=True).copy()
y_test = y.iloc[ve:].reset_index(drop=True).copy()

with open('/mnt/agents/output/lendingclub_project/models/preprocessor_fe.pkl','rb') as f:
    preprocessor_fe = pickle.load(f)

X_train_fe = preprocessor_fe.transform(X_train)
X_val_fe = preprocessor_fe.transform(X_val)
X_test_fe = preprocessor_fe.transform(X_test)
test_fe = preprocessor_fe.transform(test_features)

def get_types(df):
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    cat = df.select_dtypes(include=['object','category']).columns.tolist()
    bin_ = [c for c in num if set(df[c].dropna().unique()).issubset({0,1})]
    con = [c for c in num if c not in bin_]
    return con, bin_, cat

cont_cols, bin_cols, cat_cols = get_types(X_train_fe)

pp = ColumnTransformer(transformers=[
    ('num', Pipeline([('im',SimpleImputer(strategy='median')),('sc',StandardScaler())]), cont_cols),
    ('bin', SimpleImputer(strategy='most_frequent'), bin_cols),
    ('cat', Pipeline([('im',SimpleImputer(strategy='constant',fill_value='Missing')),
        ('oh',OneHotEncoder(handle_unknown='ignore',sparse_output=False,drop='first'))]), cat_cols)
], remainder='drop')

X_train_p = pp.fit_transform(X_train_fe)
X_val_p = pp.transform(X_val_fe)
X_test_p = pp.transform(X_test_fe)
test_p = pp.transform(test_fe)

NF = X_train_p.shape[1]
print(f"Features: {NF}")

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class FTTransformer(nn.Module):
    """Lightweight FT-Transformer for CPU training."""
    def __init__(self, n_feat, d_token=16, n_blocks=2, n_heads=4, d_h=64, drop=0.1):
        super().__init__()
        self.d_token = d_token
        # Use a shared embedding lookup table approach for efficiency
        self.num_embed = nn.Linear(n_feat, d_token)
        self.cls = nn.Parameter(torch.randn(1, 1, d_token))
        enc = nn.TransformerEncoderLayer(d_model=d_token, nhead=n_heads, dim_feedforward=d_h,
                                         dropout=drop, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_blocks)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, d_h), nn.GELU(), nn.Dropout(drop), nn.Linear(d_h, 1))
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
    def forward(self, x):
        B = x.shape[0]
        # Embed all features together (more efficient than per-feature)
        tok = self.num_embed(x).unsqueeze(1)  # [B, 1, d_token]
        cls = self.cls.expand(B, -1, -1)
        x_t = torch.cat([cls, tok], dim=1)
        out = self.transformer(x_t)
        return self.head(out[:, 0, :]).squeeze(-1)

class TabMLP(nn.Module):
    def __init__(self, inp, hid=[256,128,64], drop=0.2):
        super().__init__()
        layers = []
        prev = inp
        for h in hid:
            layers += [nn.Linear(prev,h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(drop)]
            prev = h
        layers.append(nn.Linear(prev,1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

Xt = torch.FloatTensor(np.array(X_train_p, dtype=np.float32))
yt = torch.FloatTensor(np.array(y_train.values, dtype=np.float32)).unsqueeze(1)
Xv = torch.FloatTensor(np.array(X_val_p, dtype=np.float32))
yv = torch.FloatTensor(np.array(y_val.values, dtype=np.float32)).unsqueeze(1)
Xte = torch.FloatTensor(np.array(X_test_p, dtype=np.float32))
yte = torch.FloatTensor(np.array(y_test.values, dtype=np.float32)).unsqueeze(1)
tst = torch.FloatTensor(np.array(test_p, dtype=np.float32))

trl = DataLoader(TensorDataset(Xt, yt), batch_size=512, shuffle=True)
vll = DataLoader(TensorDataset(Xv, yv), batch_size=1024, shuffle=False)
tel = DataLoader(TensorDataset(Xte, yte), batch_size=1024, shuffle=False)

def train(m, trl, vll, epochs=30, lr=1e-3, wd=1e-4, patience=8):
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=4)
    crit = nn.MSELoss()
    best, bst, pc = float('inf'), None, 0
    for e in range(epochs):
        m.train()
        for xb, yb in trl:
            opt.zero_grad()
            p = m(xb)
            loss = crit(p, yb.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        m.eval()
        vp, vt = [], []
        with torch.no_grad():
            for xb, yb in vll:
                vp.extend(m(xb).numpy())
                vt.extend(yb.squeeze().numpy())
        vrmse = np.sqrt(np.mean((np.array(vt)-np.array(vp))**2))
        sch.step(vrmse)
        if vrmse < best:
            best, bst, pc = vrmse, {k:v.cpu().clone() for k,v in m.state_dict().items()}, 0
        else:
            pc += 1
        if (e+1) % 5 == 0:
            print(f"  Epoch {e+1:3d}: val_rmse={vrmse:.4f}, best={best:.4f}")
        if pc >= patience:
            print(f"  Early stop at epoch {e+1}")
            break
    m.load_state_dict(bst)
    return m, best

def ev(m, loader):
    m.eval()
    p, t = [], []
    with torch.no_grad():
        for xb, yb in loader:
            p.extend(m(xb).numpy())
            t.extend(yb.squeeze().numpy())
    p, t = np.array(p), np.array(t)
    return {'RMSE':np.sqrt(np.mean((t-p)**2)),'MAE':np.mean(np.abs(t-p)),'MAPE':np.mean(np.abs((t-p)/t))*100,
            'R2':1-np.sum((t-p)**2)/np.sum((t-t.mean())**2),'predictions':p}

# Train FT-Transformer
print("\n" + "="*60 + "\nFT-TRANSFORMER\n" + "="*60)
ft = FTTransformer(NF, d_token=16, n_blocks=2, n_heads=4, d_h=64, drop=0.1)
print(f"Params: {sum(p.numel() for p in ft.parameters()):,}")
ft, _ = train(ft, trl, vll, epochs=30, lr=5e-3, wd=1e-4, patience=8)
fte = ev(ft, tel)
print(f"TEST: RMSE={fte['RMSE']:.4f}, MAE={fte['MAE']:.4f}, MAPE={fte['MAPE']:.2f}%, R2={fte['R2']:.4f}")

# Train TabMLP
print("\n" + "="*60 + "\nTAB-MLP\n" + "="*60)
mlp = TabMLP(NF, hid=[256,128,64], drop=0.2)
print(f"Params: {sum(p.numel() for p in mlp.parameters()):,}")
mlp, _ = train(mlp, trl, vll, epochs=30, lr=1e-3, wd=1e-4, patience=8)
mpe = ev(mlp, tel)
print(f"TEST: RMSE={mpe['RMSE']:.4f}, MAE={mpe['MAE']:.4f}, MAPE={mpe['MAPE']:.2f}%, R2={mpe['R2']:.4f}")

# Save
pickle.dump({'ft': fte, 'mlp': mpe}, open('/mnt/agents/output/lendingclub_project/models/nn_results.pkl','wb'))
torch.save(ft.state_dict(), '/mnt/agents/output/lendingclub_project/models/model_ft.pt')
torch.save(mlp.state_dict(), '/mnt/agents/output/lendingclub_project/models/model_mlp.pt')

# Test predictions
tstl = DataLoader(tst, batch_size=1024, shuffle=False)
ft.eval(); mlp.eval()
ftp, mlpp = [], []
with torch.no_grad():
    for xb in tstl:
        ftp.extend(ft(xb).numpy())
        mlpp.extend(mlp(xb).numpy())
np.save('/mnt/agents/output/lendingclub_project/models/ft_test_preds.npy', np.array(ftp))
np.save('/mnt/agents/output/lendingclub_project/models/mlp_test_preds.npy', np.array(mlpp))
print("\nAll saved!")
