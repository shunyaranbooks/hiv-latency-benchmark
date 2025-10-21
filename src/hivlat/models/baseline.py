from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

def _prep_Xy(df: pd.DataFrame, labels: pd.Series, classes: list[str]):
    # df: genes x cells ; we need cells x genes
    X = df.T.values
    y = pd.Categorical(labels, categories=classes)
    return X, y.codes, list(y.categories)

def train_model(df_train: pd.DataFrame, labels_train: pd.Series, model_type='logistic', random_state=42, max_iter=800):
    classes = sorted(labels_train.unique().tolist())
    X, y, classes = _prep_Xy(df_train, labels_train, classes)
    scaler = StandardScaler(with_mean=False)
    Xs = scaler.fit_transform(X)
    if model_type == 'gbm':
        clf = GradientBoostingClassifier(random_state=random_state)
    else:
        clf = LogisticRegression(max_iter=max_iter, random_state=random_state, multi_class='auto')
    clf.fit(Xs, y)
    return {'model': clf, 'scaler': scaler, 'classes': classes}

def predict_proba(bundle, df: pd.DataFrame):
    X = df.T.values
    Xs = bundle['scaler'].transform(X)
    proba = bundle['model'].predict_proba(Xs)
    return proba  # cells x classes

def eval_multiclass(bundle, df: pd.DataFrame, labels: pd.Series):
    from sklearn.preprocessing import label_binarize
    classes = bundle['classes']
    proba = predict_proba(bundle, df)
    y_true = pd.Categorical(labels, categories=classes).codes
    # AUROC/AUPRC (macro one-vs-rest)
    Y_bin = label_binarize(y_true, classes=list(range(len(classes))))
    aurocs, auprcs = [], []
    for k in range(len(classes)):
        aurocs.append(roc_auc_score(Y_bin[:,k], proba[:,k]))
        auprcs.append(average_precision_score(Y_bin[:,k], proba[:,k]))
    return {
        'auroc_macro_ovr': float(np.mean(aurocs)),
        'auprc_macro_ovr': float(np.mean(auprcs))
    }, proba, y_true

def save(bundle, path):
    joblib.dump(bundle, path)

def load(path):
    return joblib.load(path)
