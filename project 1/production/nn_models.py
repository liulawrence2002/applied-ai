"""Deep-learning fitters: tabular FT-Transformer + capacity-boosted MLP.

Both train on the same numeric/one-hot frame produced by `to_numeric_frame`.
Each fit returns (val_pred, test_pred, fitted_model_state) and is structurally
orthogonal to GBDTs (no tree splits → uncorrelated errors).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from production.encoders import to_numeric_frame

torch.set_num_threads(min(16, torch.get_num_threads()))
torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------
# FT-Transformer-lite (single shared linear embedder, CLS aggregation)
# ----------------------------------------------------------------------------
class FTTransformerLite(nn.Module):
    def __init__(self, n_features, d_token=32, n_blocks=3, n_heads=4, d_hidden=128, dropout=0.15):
        super().__init__()
        self.embed = nn.Linear(n_features, d_token)
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_hidden,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token), nn.Linear(d_token, d_hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_hidden, 1),
        )
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        b = x.shape[0]
        tok = self.embed(x).unsqueeze(1)  # [B, 1, d]
        x = torch.cat([self.cls.expand(b, -1, -1), tok], dim=1)
        return self.head(self.transformer(x)[:, 0, :]).squeeze(-1)


class TabMLPDeep(nn.Module):
    """Higher-capacity MLP with residual blocks and SE-style gating."""
    def __init__(self, n_in, dims=(256, 256, 128, 64), dropout=0.2):
        super().__init__()
        layers = []
        prev = n_in
        for h in dims:
            block = nn.Sequential(
                nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout),
            )
            layers.append(block)
            prev = h
        self.blocks = nn.ModuleList(layers)
        self.skip_proj = nn.ModuleList([
            nn.Linear(n_in if i == 0 else dims[i - 1], dims[i])
            for i in range(len(dims))
        ])
        self.head = nn.Linear(prev, 1)

    def forward(self, x):
        h = x
        for block, skip in zip(self.blocks, self.skip_proj):
            h_new = block(h)
            h = h_new + skip(h)
        return self.head(h).squeeze(-1)


# ----------------------------------------------------------------------------
# Training loop helpers
# ----------------------------------------------------------------------------
def _train_torch_model(model, Xtr, ytr, Xva, yva, *,
                       lr=2e-3, wd=1e-4, epochs=50, batch_size=512,
                       patience=10, verbose=False):
    """Generic Adam-W training loop with ReduceLROnPlateau + early stop."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
    crit = nn.MSELoss()

    Xtr_t = torch.FloatTensor(np.array(Xtr, dtype=np.float32))
    ytr_t = torch.FloatTensor(np.array(ytr, dtype=np.float32))
    Xva_t = torch.FloatTensor(np.array(Xva, dtype=np.float32))
    yva_t = torch.FloatTensor(np.array(yva, dtype=np.float32))
    trl = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True)
    vll = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=1024, shuffle=False)

    best, best_state, pc = float("inf"), None, 0
    for e in range(epochs):
        model.train()
        for xb, yb in trl:
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            preds = []
            for xb, _ in vll:
                preds.append(model(xb).numpy())
            preds = np.concatenate(preds)
        rmse = float(np.sqrt(((yva - preds) ** 2).mean()))
        sched.step(rmse)
        if rmse < best:
            best, best_state = rmse, {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
        if verbose and (e + 1) % 5 == 0:
            print(f"      epoch {e+1:3d}  val_rmse={rmse:.4f}  best={best:.4f}", flush=True)
        if pc >= patience:
            if verbose:
                print(f"      early stop @ epoch {e+1}", flush=True)
            break

    model.load_state_dict(best_state)
    return model, best


def _predict_torch(model, X, batch_size=1024):
    model.eval()
    Xt = torch.FloatTensor(np.array(X, dtype=np.float32))
    loader = DataLoader(TensorDataset(Xt), batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb).numpy())
    return np.concatenate(preds)


def _prep_numeric(Xtr, Xva, Xte):
    Xtr_n = to_numeric_frame(Xtr)
    Xva_n = to_numeric_frame(Xva).reindex(columns=Xtr_n.columns, fill_value=0)
    Xte_n = to_numeric_frame(Xte).reindex(columns=Xtr_n.columns, fill_value=0)
    imp = SimpleImputer(strategy="median").fit(Xtr_n.values)
    sc = StandardScaler().fit(imp.transform(Xtr_n.values))
    Xtr_s = sc.transform(imp.transform(Xtr_n.values))
    Xva_s = sc.transform(imp.transform(Xva_n.values))
    Xte_s = sc.transform(imp.transform(Xte_n.values))
    return Xtr_s, Xva_s, Xte_s, (imp, sc, list(Xtr_n.columns))


def fit_ft_transformer(Xtr, ytr, Xva, yva, Xte, seed, Xval_extra=None):
    """Returns (val_pred_on_Xva, test_pred_on_Xte, extra_pred_on_Xval_or_None, prep)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr_s, Xva_s, Xte_s, prep = _prep_numeric(Xtr, Xva, Xte)
    model = FTTransformerLite(n_features=Xtr_s.shape[1], d_token=32, n_blocks=3,
                              n_heads=4, d_hidden=128, dropout=0.15)
    model, _ = _train_torch_model(model, Xtr_s, ytr, Xva_s, yva,
                                   lr=3e-3, wd=1e-4, epochs=40, batch_size=512, patience=8)
    val_pred = _predict_torch(model, Xva_s)
    test_pred = _predict_torch(model, Xte_s)
    extra_pred = None
    if Xval_extra is not None:
        imp, sc, cols = prep
        Xval_n = to_numeric_frame(Xval_extra).reindex(columns=cols, fill_value=0)
        Xval_s = sc.transform(imp.transform(Xval_n.values))
        extra_pred = _predict_torch(model, Xval_s)
    return val_pred, test_pred, extra_pred, prep


def fit_tab_mlp_deep(Xtr, ytr, Xva, yva, Xte, seed, Xval_extra=None):
    """Returns (val_pred_on_Xva, test_pred_on_Xte, extra_pred_on_Xval_or_None, prep)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr_s, Xva_s, Xte_s, prep = _prep_numeric(Xtr, Xva, Xte)
    model = TabMLPDeep(n_in=Xtr_s.shape[1], dims=(256, 256, 128, 64), dropout=0.2)
    model, _ = _train_torch_model(model, Xtr_s, ytr, Xva_s, yva,
                                   lr=2e-3, wd=1e-4, epochs=50, batch_size=512, patience=10)
    val_pred = _predict_torch(model, Xva_s)
    test_pred = _predict_torch(model, Xte_s)
    extra_pred = None
    if Xval_extra is not None:
        imp, sc, cols = prep
        Xval_n = to_numeric_frame(Xval_extra).reindex(columns=cols, fill_value=0)
        Xval_s = sc.transform(imp.transform(Xval_n.values))
        extra_pred = _predict_torch(model, Xval_s)
    return val_pred, test_pred, extra_pred, prep
