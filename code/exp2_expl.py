"""Experiment 2: explanation properties.
(a) Local monotonicity violations: fraction of test records whose predicted risk falls when a
    risk factor with a clinical direction prior is increased by a fixed clinical step.
(b) Attribution stability: 10 bootstrap refits of each model; per-record Spearman correlation
    and top-3 Jaccard of feature attributions across refits (GAMA: exact contributions;
    LightGBM: TreeSHAP; EBM: per-term local scores; LR: coefficient x centred value).
(c) Additivity error of each explanation against the model log-odds."""
import pickle, numpy as np, pandas as pd, shap, itertools, time
from scipy.stats import spearmanr
from scipy.special import logit
from common import *

X, y = load()
E1 = pickle.load(open('/home/claude/p2/results/exp1.pkl', 'rb'))
folds = E1['folds'][0]
STEP = {'age': 5, 'sysBP': 10, 'totChol': 20, 'glucose': 10, 'BMI': 2, 'cigsPerDay': 5}
import sys
MODELS = sys.argv[1:] or ['LR', 'LightGBM', 'LightGBM-mono', 'EBM', 'GAMA']
res = dict(viol=[], stab={}, add={})


def fitm(name, k, Xtr, ytr):
    Cls, _ = FAMILIES[name]
    return Cls(**E1['hp'][(0, name)][k]).fit(Xtr, ytr)


# (a) monotonicity violations on all 5 test folds
for k, (tr, te) in enumerate(folds):
    tr, te = np.array(tr), np.array(te)
    for name in MODELS:
        m = fitm(name, k, X.iloc[tr], y[tr])
        Xt = X.iloc[te]
        base = m.predict_proba(Xt)[:, 1]
        for f, s in STEP.items():
            Xs = Xt.copy(); Xs[f] = Xs[f] + s
            if f == 'cigsPerDay':
                Xs['currentSmoker'] = np.where(Xs['cigsPerDay'] > 0, 1, Xs['currentSmoker'])
            ps = m.predict_proba(Xs)[:, 1]
            ok = Xt[f].notna().values
            res['viol'].append(dict(fold=k, model=name, feat=f, n=int(ok.sum()),
                                    v=int(((ps - base) < -1e-9)[ok].sum()),
                                    vbig=int(((ps - base) < -0.01)[ok].sum())))
    print('viol fold', k, flush=True)


def attributions(name, m, Xe):
    if name == 'GAMA':
        base, C, names = m.contributions(Xe)
        A = pd.DataFrame(0.0, index=Xe.index, columns=FEATS)
        for j, n in enumerate(names):
            if n in FEATS: A[n] += C[:, j]
            elif '×' in n:
                a, b = n.split('×'); A[a] += C[:, j] / 2; A[b] += C[:, j] / 2
            elif n.endswith('_missing'): A[n[:-8]] += C[:, j]
        return A.values, base + C.sum(1)
    if name.startswith('LightGBM'):
        ex = shap.TreeExplainer(m.m)
        sv = ex.shap_values(Xe)
        sv = sv[1] if isinstance(sv, list) else sv
        return sv, ex.expected_value + sv.sum(1) if np.ndim(ex.expected_value) == 0 else None
    if name == 'EBM':
        ebm = m.m; T = ebm.eval_terms(Xe)
        A = np.zeros((len(Xe), len(FEATS)))
        for j, feats in enumerate(ebm.term_features_):
            for fi in feats:
                A[:, fi] += T[:, j] / len(feats)
        return A, ebm.intercept_[0] + T.sum(1)
    if name == 'LR':
        pipe = m.m; imp, sc, lr = pipe[0], pipe[1], pipe[2]
        Z = sc.transform(imp.transform(Xe)); W = Z * lr.coef_[0]
        A = W[:, :len(FEATS)].copy()
        mi = imp.indicator_.features_
        for j, fi in enumerate(mi):
            A[:, fi] += W[:, len(FEATS) + j]
        return A, lr.intercept_[0] + W.sum(1)


# (b) stability + (c) additivity on fold 0, 100 random complete-case test records
tr, te = np.array(folds[0][0]), np.array(folds[0][1])
rng = np.random.default_rng(11)
cc = [i for i in te if not X.iloc[i].isna().any()]
ev = rng.choice(cc, 100, replace=False); Xe = X.iloc[ev]
B = 10
for name in MODELS:
    As = []
    t0 = time.time()
    for b in range(B):
        bi = rng.choice(tr, len(tr), replace=True)
        m = fitm(name, 0, X.iloc[bi], y[bi])
        A, zhat = attributions(name, m, Xe)
        if b == 0:
            z = logit(np.clip(m.predict_proba(Xe)[:, 1], 1e-12, 1 - 1e-12))
            res['add'][name] = float(np.max(np.abs(zhat - z))) if zhat is not None else None
        As.append(A)
    As = np.array(As)  # B x n x d
    rho, jac, signflip = [], [], []
    for r in range(len(ev)):
        rr, jj = [], []
        for a, c in itertools.combinations(range(B), 2):
            rr.append(spearmanr(As[a, r], As[c, r]).correlation)
            ta = set(np.argsort(-np.abs(As[a, r]))[:3]); tc = set(np.argsort(-np.abs(As[c, r]))[:3])
            jj.append(len(ta & tc) / len(ta | tc))
        rho.append(np.nanmean(rr)); jac.append(np.mean(jj))
        top = np.argsort(-np.abs(As[:, r]).mean(0))[:3]
        signflip.append(np.mean([len(set(np.sign(As[:, r, f])) - {0}) > 1 for f in top]))
    res['stab'][name] = dict(rho=np.array(rho), jac=np.array(jac), flip=np.array(signflip))
    print('stab', name, round(np.mean(rho), 3), round(np.mean(jac), 3), round(time.time() - t0, 1), flush=True)
import os
P = '/home/claude/p2/results/exp2.pkl'
if os.path.exists(P) and sys.argv[1:]:
    old = pickle.load(open(P, 'rb'))
    old['viol'] = [v for v in old['viol'] if v['model'] not in MODELS] + res['viol']
    old['stab'].update(res['stab']); old['add'].update(res['add']); res = old
pickle.dump(res, open(P, 'wb'))
