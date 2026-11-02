"""Real-time benchmark. For every complete-case test record of repeat 1 (five outer folds),
time each stage of the CertiRisk pipeline on a single record, sequentially, as a deployed
service would run it. Also time per-record explanations from TreeSHAP, KernelSHAP and LIME
(on LightGBM) and EBM local explanations for comparison."""
import pickle, time, json, datetime, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from scipy.special import expit
from common import *
from emc import EMC
from conformal import Mondrian, groups, status
from narrative import payload, compile_narrative, verify_full
warnings.filterwarnings('ignore')
import shap
from lime.lime_tabular import LimeTabularExplainer

X, y = load()
E1 = pickle.load(open('/home/claude/p3/results/exp_ext.pkl', 'rb'))
folds = E1['folds'][0]
LOG = open('/home/claude/p3/logs/exp_rt.log', 'a')
def log(*a):
    s = datetime.datetime.utcnow().strftime('%H:%M:%S') + ' ' + ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

pc = time.perf_counter
rows, cmp_rows = [], []
rng = np.random.default_rng(5)
for k, (tr, te) in enumerate(folds):
    tr, te = np.array(tr), np.array(te)
    trp, sel = train_test_split(tr, test_size=0.25, stratify=y[tr], random_state=300 + k)
    m = GAMA(lam=10).fit(X.iloc[trp], y[trp])
    psel = m.predict_proba(X.iloc[sel])[:, 1]; thr = threshold_for_recall(y[sel], psel, 0.75)
    cp = Mondrian(0.10).fit(psel, y[sel], groups(X.iloc[sel]))
    emc = EMC(m, X.iloc[trp])
    cc = [i for i in te if not X.iloc[i].isna().any()]
    for i in cc:
        xr = X.iloc[[i]]; x = X.iloc[i].to_dict(); t = {}
        a = pc(); p = float(m.predict_proba(xr)[0, 1]); t['predict'] = pc() - a
        a = pc(); base, Cc, names = m.contributions(xr); t['explain'] = pc() - a
        a = pc(); st = status(cp.predict_sets(np.array([p]), groups(xr)))[0]; t['conformal'] = pc() - a
        a = pc(); cf = None
        if p >= thr:
            sols, _ = emc.search(X.iloc[i], thr - 0.001); cf = sols[0] if sols else None
        t['counterfactual'] = pc() - a
        a = pc(); P = payload(m, x, f'F{k+1}_R{i:04d}', thr, conformal=st, cf=cf); txt = compile_narrative(m, P); t['compile'] = pc() - a
        a = pc(); rs = verify_full(txt, m, x, thr, conformal=st); t['verify'] = pc() - a
        t['total'] = sum(t.values()); t.update(fold=k, i=int(i), positive=bool(p >= thr), passed=len(rs) == 0)
        rows.append(t)
    log('fold', k + 1, 'records', len(cc), 'median total ms', round(1000 * np.median([r['total'] for r in rows if r['fold'] == k]), 1))
    # ---- explanation-latency comparison on 50 random complete-case records of this fold ----
    hp = E1['hp'][(0, 'LightGBM')][k]; lg = LGBM(**hp).fit(X.iloc[trp], y[trp])
    ebm = EBM().fit(X.iloc[trp], y[trp])
    Xtr_f = X.iloc[trp].fillna(X.iloc[trp].median())
    tree = shap.TreeExplainer(lg.m)
    bg = shap.sample(Xtr_f, 50, random_state=0)
    kern = shap.KernelExplainer(lambda Z: lg.m.predict_proba(pd.DataFrame(Z, columns=FEATS))[:, 1], bg)
    lime_ex = LimeTabularExplainer(Xtr_f.values, feature_names=FEATS, class_names=['no', 'CHD'], discretize_continuous=True, random_state=0)
    for i in rng.choice(cc, 50, replace=False):
        xr = X.iloc[[i]]; xv = xr.fillna(X.iloc[trp].median())
        d = {}
        a = pc(); m.contributions(xr); d['GAMA exact contributions'] = pc() - a
        a = pc(); ebm.m.eval_terms(xr); d['EBM term scores'] = pc() - a
        a = pc(); tree.shap_values(xv); d['TreeSHAP (LightGBM)'] = pc() - a
        a = pc(); kern.shap_values(xv.values, nsamples=500, silent=True); d['KernelSHAP (LightGBM, 500 samples)'] = pc() - a
        a = pc(); lime_ex.explain_instance(xv.values[0], lambda Z: lg.m.predict_proba(pd.DataFrame(Z, columns=FEATS)), num_features=5, num_samples=5000); d['LIME (LightGBM, 5,000 samples)'] = pc() - a
        d.update(fold=k); cmp_rows.append(d)
    log('fold', k + 1, 'explainer timing done')
pickle.dump(dict(rows=rows, cmp=cmp_rows), open('/home/claude/p3/results/exp_rt.pkl', 'wb'))
log('DONE', len(rows), 'records timed')
