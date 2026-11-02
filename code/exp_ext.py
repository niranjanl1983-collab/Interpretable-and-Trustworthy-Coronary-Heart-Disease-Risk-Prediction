"""Extended comparison: GAMA vs 13 classifiers under repeated nested cross-validation.
Same outer/inner fold seeds as the main study. Logs every fit with a timestamp."""
import pickle, time, sys, json, datetime, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from common import *
warnings.filterwarnings('ignore')

X, y = load()
LOG = open('/home/claude/p3/logs/exp_ext.log', 'a')
def log(*a):
    s = datetime.datetime.utcnow().strftime('%H:%M:%S') + ' ' + ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


class SK:
    """Wraps a scikit-learn style estimator with median imputation (+indicators) and scaling."""
    def __init__(self, make, scale=True, **kw):
        self.make, self.scale, self.kw = make, scale, kw
    def fit(self, X, y):
        steps = [SimpleImputer(strategy='median', add_indicator=True)] + ([StandardScaler()] if self.scale else []) + [self.make(**self.kw)]
        self.m = make_pipeline(*steps).fit(X, y); return self
    def predict_proba(self, X):
        return self.m.predict_proba(X)


def mk(make, scale=True):
    return lambda **kw: SK(make, scale, **kw)


EXT = {
    'Logistic regression': (LR, [dict(C=c) for c in (0.01, 0.1, 1.0)]),
    'Decision tree': (mk(DecisionTreeClassifier, False), [dict(max_depth=d, min_samples_leaf=20, random_state=0) for d in (3, 5, 7)]),
    'Random forest': (mk(RandomForestClassifier, False), [dict(n_estimators=300, min_samples_leaf=l, max_features='sqrt', n_jobs=2, random_state=0) for l in (5, 20)]),
    'Extra trees': (mk(ExtraTreesClassifier, False), [dict(n_estimators=300, min_samples_leaf=l, n_jobs=2, random_state=0) for l in (5, 20)]),
    'AdaBoost': (mk(AdaBoostClassifier, False), [dict(n_estimators=n, learning_rate=0.1, random_state=0) for n in (100, 300)]),
    'Gradient boosting': (mk(GradientBoostingClassifier, False), [dict(n_estimators=n, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0) for n in (100, 300)]),
    'XGBoost': (mk(XGBClassifier, False), [dict(n_estimators=n, max_depth=d, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                                                 reg_lambda=1.0, n_jobs=2, random_state=0, verbosity=0) for n in (200, 400) for d in (2, 4)]),
    'LightGBM': FAMILIES['LightGBM'],
    'SVM (RBF)': (mk(SVC), [dict(C=c, gamma=g, probability=True, random_state=0) for c in (0.1, 1.0, 10.0) for g in ('scale', 0.01)]),
    'K-nearest neighbours': (mk(KNeighborsClassifier), [dict(n_neighbors=k, weights='distance') for k in (25, 75)]),
    'Gaussian naive Bayes': (mk(GaussianNB), [dict()]),
    'MLP': (mk(MLPClassifier), [dict(hidden_layer_sizes=h, alpha=a, early_stopping=True, max_iter=500, random_state=0) for h in ((16,), (32, 16)) for a in (1e-2, 1.0)]),
    'EBM': FAMILIES['EBM'],
    'GAMA (proposed)': FAMILIES['GAMA'],
}

REPEATS, OUTER, INNER = 2, 5, 3
P = '/home/claude/p3/results/exp_ext.pkl'
try:
    res = pickle.load(open(P, 'rb'))
except FileNotFoundError:
    res = {'oof': {}, 'thr': {}, 'hp': {}, 'fit_s': {}, 'pred_ms': {}, 'fold_auc': {}, 'folds': {}}
names = sys.argv[1:] or list(EXT)
log('START', 'records', len(y), 'events', int(y.sum()), 'models', len(names))
from sklearn.metrics import roc_auc_score
for r in range(REPEATS):
    outer = StratifiedKFold(OUTER, shuffle=True, random_state=100 + r)
    folds = list(outer.split(X, y)); res['folds'][r] = [(a.tolist(), b.tolist()) for a, b in folds]
    for name in names:
        if (r, name) in res['oof']:
            continue
        Cls, grid = EXT[name]
        oof = np.zeros(len(y)); thr = np.zeros(len(y)); hps, fits, preds, fa = [], [], [], []
        for k, (tr, te) in enumerate(folds):
            Xtr, ytr = X.iloc[tr], y[tr]
            inner = StratifiedKFold(INNER, shuffle=True, random_state=200 + 10 * r + k)
            best = None
            for hp in grid:
                ip = np.zeros(len(tr))
                for itr, iva in inner.split(Xtr, ytr):
                    ip[iva] = Cls(**hp).fit(Xtr.iloc[itr], ytr[itr]).predict_proba(Xtr.iloc[iva])[:, 1]
                ll = log_loss(ytr, np.clip(ip, 1e-6, 1 - 1e-6))
                if best is None or ll < best[0]:
                    best = (ll, hp, ip)
            _, hp, ip = best
            t = threshold_for_recall(ytr, ip, 0.75)
            t0 = time.perf_counter(); m = Cls(**hp).fit(Xtr, ytr); fits.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); pte = m.predict_proba(X.iloc[te])[:, 1]; preds.append(1000 * (time.perf_counter() - t0) / len(te))
            oof[te] = pte; thr[te] = t; hps.append(hp); fa.append(roc_auc_score(y[te], pte))
        res['oof'][(r, name)] = oof; res['thr'][(r, name)] = thr; res['hp'][(r, name)] = hps
        res['fit_s'][(r, name)] = fits; res['pred_ms'][(r, name)] = preds; res['fold_auc'][(r, name)] = fa
        pickle.dump(res, open(P, 'wb'))
        log(f'repeat {r+1}', f'{name:22s}', 'AUC', round(roc_auc_score(y, oof), 4), 'fold AUCs', [round(v, 3) for v in fa],
            'fit s', round(float(np.mean(fits)), 2), 'hp', hps[0])
log('DONE')
