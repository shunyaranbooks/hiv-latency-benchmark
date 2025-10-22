from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

def _prep_Xy(df: pd.DataFrame, labels: pd.Series, classes: list[str]):
    # df: genes x cells ; model expects cells x genes
    X = df.T.values
    y = pd.Categorical(labels, categories=classes)
    return X, y.codes, list(y.categories)

def train_model(df_train: pd.DataFrame, labels_train: pd.Series, model_type='logistic', random_state=42, max_iter=800):
    # classes sorted for stable order
    classes = sorted(labels_train.unique().tolist())
    # remember the exact gene order used for training
    genes = df_train.index.tolist()

    X, y, classes = _prep_Xy(df_train, labels_train, classes)
    scaler = StandardScaler(with_mean=False)
    Xs = scaler.fit_transform(X)

    if model_type == 'gbm':
        clf = GradientBoostingClassifier(random_state=random_state)
    else:
        clf = LogisticRegression(max_iter=max_iter, random_state=random_state)

    clf.fit(Xs, y)
    return {'model': clf, 'scaler': scaler, 'classes': classes, 'genes': genes}

def _align_to_genes(df: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    """
    Reindex df (genes x cells) to the training gene order; fill missing genes with 0,
    drop any extra genes silently.
    """
    if df.index.equals(pd.Index(genes)):
        return df
    # reindex introduces NaNs for missing genes (fill with 0)
    aligned = df.reindex(genes)
    return aligned.fillna(0.0)

def predict_proba(bundle, df: pd.DataFrame):
    genes = bundle.get('genes', None)
    if genes is not None:
        df = _align_to_genes(df, genes)
    X_df = df.T  # keep DataFrame so feature names are preserved
    Xs = bundle['scaler'].transform(X_df)
    proba = bundle['model'].predict_proba(Xs)
    return proba  # shape: cells x classes

def eval_multiclass(bundle, df: pd.DataFrame, labels: pd.Series):
    from sklearn.preprocessing import label_binarize
    classes = bundle['classes']
    proba = predict_proba(bundle, df)
    y_true = pd.Categorical(labels, categories=classes).codes
    Y_bin = label_binarize(y_true, classes=list(range(len(classes))))
    aurocs, auprcs = [], []
    for k in range(len(classes)):
        # Skip classes absent in y_true to avoid metrics errors
        mask_valid = Y_bin[:, k] >= 0
        try:
            aurocs.append(roc_auc_score(Y_bin[:, k], proba[:, k]))
            auprcs.append(average_precision_score(Y_bin[:, k], proba[:, k]))
        except Exception:
            pass
    out = {
        'auroc_macro_ovr': float(np.mean(aurocs)) if aurocs else float('nan'),
        'auprc_macro_ovr': float(np.mean(auprcs)) if auprcs else float('nan')
    }
    return out, proba, y_true

def save(bundle, path):
    joblib.dump(bundle, path)

def load(path):
    return joblib.load(path)
