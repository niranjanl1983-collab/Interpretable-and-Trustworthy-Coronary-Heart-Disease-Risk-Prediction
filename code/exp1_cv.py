"""Experiment 1: repeated nested cross-validation of GAMA vs baselines and ablations."""
import json, time, pickle, sys
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from common import *

X, y = load()
REPEATS, OUTER, INNER = 2, 5, 3
res = {'oof': {}, 'thr': {}, 'hp': {}, 'folds': {}, 'time': {}}
for r in range(REPEATS):
    outer = StratifiedKFold(OUTER, shuffle=True, random_state=100 + r)
    folds = list(outer.split(X, y)); res['folds'][r] = [(tr.tolist(), te.tolist()) for tr, te in folds]
    for name, (Cls, grid) in FAMILIES.items():
        oof = np.zeros(len(y)); thr = np.zeros(len(y)); hps = []; t0 = time.time()
        for k, (tr, te) in enumerate(folds):
            Xtr, ytr = X.iloc[tr], y[tr]
            inner = StratifiedKFold(INNER, shuffle=True, random_state=200 + 10 * r + k)
            best = None
            for hp in grid:
                ip = np.zeros(len(tr))
                for itr, iva in inner.split(Xtr, ytr):
                    ip[iva] = Cls(**hp).fit(Xtr.iloc[itr], ytr[itr]).predict_proba(Xtr.iloc[iva])[:, 1]
                ll = log_loss(ytr, ip)
                if best is None or ll < best[0]:
                    best = (ll, hp, ip)
            _, hp, ip = best
            t = threshold_for_recall(ytr, ip, 0.75)          # locked before touching the outer fold
            m = Cls(**hp).fit(Xtr, ytr)
            oof[te] = m.predict_proba(X.iloc[te])[:, 1]; thr[te] = t; hps.append(hp)
        res['oof'][(r, name)] = oof; res['thr'][(r, name)] = thr; res['hp'][(r, name)] = hps
        res['time'][(r, name)] = time.time() - t0
        print(r, name, round(time.time() - t0, 1), 's', flush=True)
pickle.dump(res, open('/home/claude/p2/results/exp1.pkl', 'wb'))
