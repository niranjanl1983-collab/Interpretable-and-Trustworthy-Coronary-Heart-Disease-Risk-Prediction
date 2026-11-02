"""Analysis of the extended comparison and the real-time benchmark."""
import pickle, json, numpy as np, pandas as pd, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, rankdata, studentized_range
from sklearn.metrics import roc_auc_score, average_precision_score
from common import load, metrics, net_benefit
from analyze1_style import *
warnings.filterwarnings('ignore')
R = '/home/claude/p3/results/'; FIG = '/home/claude/p3/figs/'
import os; os.makedirs(FIG, exist_ok=True)
X, y = load()
E = pickle.load(open(R + 'exp_ext.pkl', 'rb'))
names = [n for (r, n) in E['oof'] if r == 0]
rng = np.random.default_rng(1); boot = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
G = 'GAMA (proposed)'; g0 = E['oof'][(0, G)]
out = {}
for n in names:
    rows = [metrics(y, E['oof'][(r, n)]) for r in range(2)]
    d = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    p0 = E['oof'][(0, n)]
    a = [roc_auc_score(y[b], p0[b]) for b in boot]
    d['AUC_lo'], d['AUC_hi'] = [float(v) for v in np.percentile(a, [2.5, 97.5])]
    if n != G:
        dd = np.array([roc_auc_score(y[b], g0[b]) - roc_auc_score(y[b], p0[b]) for b in boot])
        d['dAUC'] = float(roc_auc_score(y, g0) - roc_auc_score(y, p0)); d['dAUC_lo'], d['dAUC_hi'] = [float(v) for v in np.percentile(dd, [2.5, 97.5])]
        d['p_boot'] = float(min(1, 2 * min((dd <= 0).mean(), (dd >= 0).mean())))
    th = []
    for r in range(2):
        p, t = E['oof'][(r, n)], E['thr'][(r, n)]; pr = p >= t
        th.append(dict(TP=(pr & (y == 1)).sum(), FP=(pr & (y == 0)).sum(), FN=(~pr & (y == 1)).sum(), TN=(~pr & (y == 0)).sum()))
    for k in th[0]: d[k] = float(np.mean([x[k] for x in th]))
    d['Recall'] = d['TP'] / (d['TP'] + d['FN']); d['Spec'] = d['TN'] / (d['TN'] + d['FP']); d['Prec'] = d['TP'] / (d['TP'] + d['FP'])
    d['F1'] = 2 * d['Prec'] * d['Recall'] / (d['Prec'] + d['Recall']); d['Acc'] = (d['TP'] + d['TN']) / len(y)
    d['BalAcc'] = (d['Recall'] + d['Spec']) / 2
    d['fit_s'] = float(np.mean([v for r in range(2) for v in E['fit_s'][(r, n)]]))
    d['pred_ms'] = float(np.mean([v for r in range(2) for v in E['pred_ms'][(r, n)]]))
    d['fold_auc'] = [v for r in range(2) for v in E['fold_auc'][(r, n)]]
    d['NB10'] = float(np.mean([net_benefit(y, E['oof'][(r, n)], 0.10) for r in range(2)]))
    d['NB20'] = float(np.mean([net_benefit(y, E['oof'][(r, n)], 0.20) for r in range(2)]))
    d['hp'] = str(E['hp'][(0, n)][0])
    out[n] = d
# Friedman test over the 10 outer-fold AUCs
M = np.array([out[n]['fold_auc'] for n in names]).T  # folds x models
fr = friedmanchisquare(*M.T)
ranks = np.mean([rankdata(-row) for row in M], axis=0)
k, Nf = len(names), M.shape[0]
q = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
CD = q * np.sqrt(k * (k + 1) / (6 * Nf))
for n, rk in zip(names, ranks): out[n]['rank'] = float(rk)
S = dict(T=out, friedman=dict(stat=float(fr.statistic), p=float(fr.pvalue), CD=float(CD), q=float(q), k=k, N=Nf),
         repro=dict(LR_prev=0.7237117328432559, LR_now=float(roc_auc_score(y, E['oof'][(0, 'Logistic regression')]))))
order = sorted(names, key=lambda n: -out[n]['AUC'])
for n in order:
    d = out[n]
    print(f"{n:22s} AUC {d['AUC']:.3f} ({d['AUC_lo']:.3f}-{d['AUC_hi']:.3f}) AP {d['AP']:.3f} Brier {d['Brier']:.4f} slope {d['CalSlope']:.2f} "
          f"rank {d['rank']:.2f} fit {d['fit_s']:.2f}s pred {d['pred_ms']:.4f}ms " + (f"dAUC {d['dAUC']:+.4f} [{d['dAUC_lo']:+.4f},{d['dAUC_hi']:+.4f}] p={d['p_boot']:.3f}" if 'dAUC' in d else ''))
print('Friedman', S['friedman'])

# figure: forest plot of AUC + mean rank
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
for i, n in enumerate(order[::-1]):
    d = out[n]; c = '#2a78d6' if n == G else '#52514e'
    ax[0].plot([d['AUC_lo'], d['AUC_hi']], [i, i], color=c, lw=1.6)
    ax[0].plot(d['AUC'], i, 'o', color=c, ms=5 if n != G else 7)
ax[0].set_yticks(range(len(order))); ax[0].set_yticklabels(order[::-1], fontsize=7.5)
ax[0].set_xlabel('ROC-AUC (95% bootstrap CI)'); ax[0].set_title('(a) Discrimination', fontsize=9, loc='left')
ax[0].axvline(out[G]['AUC'], color='#2a78d6', lw=0.8, ls=':')
ro = sorted(names, key=lambda n: out[n]['rank'])
for i, n in enumerate(ro[::-1]):
    c = '#2a78d6' if n == G else '#8a8984'
    ax[1].barh(i, out[n]['rank'], color=c, height=0.6)
ax[1].set_yticks(range(len(ro))); ax[1].set_yticklabels(ro[::-1], fontsize=7.5)
best = min(out[n]['rank'] for n in names)
ax[1].axvline(best + CD, color='#e34948', lw=1, ls='--')
ax[1].text(best + CD + 0.2, len(ro) - 1.2, f'best rank + CD\n(CD = {CD:.2f})', fontsize=6.5, color='#e34948')
ax[1].set_xlabel('Mean rank (lower = better)'); ax[1].set_title('(b) Friedman ranks', fontsize=9, loc='left')
plt.tight_layout(); plt.savefig(FIG + 'fig_ext.png'); plt.close()

# ---------------- real-time ----------------
RT = pickle.load(open(R + 'exp_rt.pkl', 'rb'))
D = pd.DataFrame(RT['rows'])
stages = ['predict', 'explain', 'conformal', 'counterfactual', 'compile', 'verify', 'total']
rt = {}
for grp, sub in [('all', D), ('positive', D[D.positive]), ('negative', D[~D.positive])]:
    rt[grp] = {s: dict(median=float(1000 * sub[s].median()), p95=float(1000 * sub[s].quantile(0.95)), max=float(1000 * sub[s].max()), mean=float(1000 * sub[s].mean())) for s in stages}
    rt[grp]['n'] = int(len(sub))
rt['passed'] = int(D.passed.sum()); rt['n'] = int(len(D))
rt['throughput_per_s'] = float(1 / D.total.mean())
Cm = pd.DataFrame(RT['cmp'])
meths = [c for c in Cm.columns if c != 'fold']
rt['explainers'] = {m: dict(median=float(1000 * Cm[m].median()), p95=float(1000 * Cm[m].quantile(0.95)), n=int(Cm[m].notna().sum())) for m in meths}
S['rt'] = rt
print(json.dumps({g: {s: round(rt[g][s]['median'], 2) for s in stages} for g in ('all', 'positive', 'negative')}, indent=0))
print({m: round(v['median'], 2) for m, v in rt['explainers'].items()}, rt['passed'], rt['n'])

fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
st = stages[:-1]
med = [rt['positive'][s]['median'] for s in st]; p95 = [rt['positive'][s]['p95'] for s in st]
ax[0].barh(range(len(st))[::-1], med, color='#2a78d6', height=0.55, label='median')
ax[0].scatter(p95, range(len(st))[::-1], color='#0b0b0b', marker='|', s=80, label='95th percentile', zorder=3)
ax[0].set_yticks(range(len(st))[::-1]); ax[0].set_yticklabels(['Prediction', 'Exact explanation', 'Conformal status', 'Counterfactual (EMC)', 'Narrative compilation', 'Certificate verification'], fontsize=7.5)
ax[0].set_xlabel('Time per record (ms), screen-positive records'); ax[0].legend(fontsize=7, loc='lower right')
ax[0].set_title('(a) Pipeline stages (proposed system)', fontsize=9, loc='left')
mm = sorted(meths, key=lambda m: rt['explainers'][m]['median'])
ax[1].barh(range(len(mm))[::-1], [rt['explainers'][m]['median'] for m in mm], color=['#2a78d6' if 'GAMA' in m else '#8a8984' for m in mm], height=0.55)
ax[1].set_xscale('log'); ax[1].set_yticks(range(len(mm))[::-1]); ax[1].set_yticklabels([m.replace(' (LightGBM', '\n(LightGBM') for m in mm], fontsize=7)
ax[1].set_xlabel('Median ms per explanation (log)'); ax[1].set_title('(b) Explanation latency', fontsize=9, loc='left')
plt.tight_layout(); plt.savefig(FIG + 'fig_rt.png'); plt.close()
json.dump(S, open(R + 'ext_rt_summary.json', 'w'), indent=1, default=float)
