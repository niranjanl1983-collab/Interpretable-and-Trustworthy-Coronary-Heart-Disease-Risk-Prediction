"""Re-run selected families of Experiment 1 with identical folds and update the results file."""
import pickle, time, sys, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from common import *
X, y = load()
res = pickle.load(open('/home/claude/p2/results/exp1.pkl', 'rb'))
for name in sys.argv[1:]:
    Cls, grid = FAMILIES[name]
    for r in range(2):
        folds = [(np.array(a), np.array(b)) for a, b in res['folds'][r]]
        oof = np.zeros(len(y)); thr = np.zeros(len(y)); hps = []; t0 = time.time()
        for k, (tr, te) in enumerate(folds):
            Xtr, ytr = X.iloc[tr], y[tr]
            inner = StratifiedKFold(3, shuffle=True, random_state=200 + 10 * r + k)
            best = None
            for hp in grid:
                ip = np.zeros(len(tr))
                for itr, iva in inner.split(Xtr, ytr):
                    ip[iva] = Cls(**hp).fit(Xtr.iloc[itr], ytr[itr]).predict_proba(Xtr.iloc[iva])[:, 1]
                ll = log_loss(ytr, ip)
                if best is None or ll < best[0]: best = (ll, hp, ip)
            _, hp, ip = best
            t = threshold_for_recall(ytr, ip, 0.75)
            oof[te] = Cls(**hp).fit(Xtr, ytr).predict_proba(X.iloc[te])[:, 1]; thr[te] = t; hps.append(hp)
        res['oof'][(r, name)] = oof; res['thr'][(r, name)] = thr; res['hp'][(r, name)] = hps; res['time'][(r, name)] = time.time() - t0
        print(r, name, hps[0], flush=True)
pickle.dump(res, open('/home/claude/p2/results/exp1.pkl', 'wb'))
