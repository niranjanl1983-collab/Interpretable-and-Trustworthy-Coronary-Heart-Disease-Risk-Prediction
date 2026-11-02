import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from scipy.special import logit, expit
import lightgbm as lgb
from interpret.glassbox import ExplainableBoostingClassifier
from gama import GAMA, MONO

DATA = '/home/claude/p3/data/framingham.csv'
FEATS = ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke',
         'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']


def load():
    d = pd.read_csv(DATA)
    return d[FEATS], d['TenYearCHD'].values


class LR:
    def __init__(self, C=0.1):
        self.C = C

    def fit(self, X, y):
        self.m = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                               LogisticRegression(C=self.C, max_iter=2000)).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


class LGBM:
    def __init__(self, leaves=8, n=300, mono=False):
        self.kw = dict(num_leaves=leaves, n_estimators=n, learning_rate=0.02, min_child_samples=40,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                       verbose=-1, random_state=0)
        if mono:
            self.kw['monotone_constraints'] = [MONO[f] for f in FEATS]

    def fit(self, X, y):
        self.m = lgb.LGBMClassifier(**self.kw).fit(X, y); return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


class EBM:
    def __init__(self, **kw):
        self.kw = kw

    def fit(self, X, y):
        self.m = ExplainableBoostingClassifier(random_state=0, outer_bags=4, n_jobs=2, **self.kw).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


# model families: name -> (constructor, hyper-parameter grid)
FAMILIES = {
    'LR': (LR, [dict(C=c) for c in (0.01, 0.1, 1.0)]),
    'LightGBM': (LGBM, [dict(leaves=l, n=n) for l in (4, 8) for n in (200, 400)]),
    'LightGBM-mono': (LGBM, [dict(leaves=l, n=n, mono=True) for l in (4, 8) for n in (200, 400)]),
    'EBM': (EBM, [dict()]),
    'GAM (unconstrained)': (GAMA, [dict(lam=l, monotone=False) for l in (1, 3, 10, 30)]),
    'GAMA (quantile knots)': (GAMA, [dict(lam=l, knots='quantile') for l in (1, 3, 10, 30)]),
    'GAMA (+ interactions)': (GAMA, [dict(lam=l, interactions=True) for l in (1, 3, 10, 30)]),
    'GAMA': (GAMA, [dict(lam=l) for l in (1, 3, 10, 30)]),
}


def cal_slope_int(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); z = logit(p)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y)
    slope = m.coef_[0, 0]
    # calibration-in-the-large: intercept with slope fixed at 1 (offset model)
    from scipy.optimize import minimize_scalar
    a = minimize_scalar(lambda a: np.sum(np.logaddexp(0, a + z) - y * (a + z))).x
    return a, slope


def ece(y, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); idx = np.clip(np.searchsorted(q, p, side='right') - 1, 0, bins - 1)
    return sum(np.abs(y[idx == b].mean() - p[idx == b].mean()) * np.mean(idx == b) for b in range(bins) if np.any(idx == b))


def threshold_for_recall(y, p, target=0.75):
    """Largest threshold whose recall >= target (maximises specificity subject to recall)."""
    order = np.sort(np.unique(p))[::-1]
    pos = y.sum()
    for t in order:
        if ((p >= t) & (y == 1)).sum() / pos >= target:
            return t
    return order[-1]


def metrics(y, p, t=None):
    a, s = cal_slope_int(y, p)
    out = dict(AUC=roc_auc_score(y, p), AP=average_precision_score(y, p), Brier=brier_score_loss(y, p),
               LogLoss=log_loss(y, p), CalInt=a, CalSlope=s, ECE=ece(y, p))
    if t is not None:
        pr = p >= t
        tp = int((pr & (y == 1)).sum()); fp = int((pr & (y == 0)).sum())
        fn = int((~pr & (y == 1)).sum()); tn = int((~pr & (y == 0)).sum())
        out.update(TP=tp, FP=fp, FN=fn, TN=tn, Recall=tp / (tp + fn), Spec=tn / (tn + fp), Prec=tp / max(tp + fp, 1))
    return out


def net_benefit(y, p, t):
    n = len(y); pr = p >= t
    return ((pr & (y == 1)).sum() / n) - ((pr & (y == 0)).sum() / n) * t / (1 - t)
