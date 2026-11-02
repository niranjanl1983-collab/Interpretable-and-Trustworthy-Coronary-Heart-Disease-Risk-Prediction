"""Tables and figures for Experiment 1 (discrimination, calibration, clinical utility, shapes)."""
import pickle, json, numpy as np, pandas as pd, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from common import *
warnings.filterwarnings('ignore')
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e',
                     'ytick.color': '#52514e', 'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6,
                     'savefig.dpi': 300, 'legend.frameon': False})
COL = {'GAMA': '#2a78d6', 'LR': '#8a8984', 'LightGBM': '#eb6834', 'EBM': '#1baf7a', 'LightGBM-mono': '#eda100',
       'GAM (unconstrained)': '#6250d6', 'GAMA (quantile knots)': '#e87ba4', 'GAMA (+ interactions)': '#008300'}
FIG = '/home/claude/p2/figs/'; RES = '/home/claude/p2/results/'
import os; os.makedirs(FIG, exist_ok=True)
X, y = load()
E = pickle.load(open(RES + 'exp1.pkl', 'rb'))
names = list(FAMILIES)
rng = np.random.default_rng(1)
B = 1000
boot = [rng.integers(0, len(y), len(y)) for _ in range(B)]
T = {}
for n in names:
    rows = [metrics(y, E['oof'][(r, n)]) for r in range(2)]
    avg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    p0 = E['oof'][(0, n)]
    aucs = [roc_auc_score(y[b], p0[b]) for b in boot]
    avg['AUC_lo'], avg['AUC_hi'] = np.percentile(aucs, [2.5, 97.5])
    # threshold metrics at fold-locked thresholds (both repeats averaged)
    th = []
    for r in range(2):
        p, t = E['oof'][(r, n)], E['thr'][(r, n)]
        pr = p >= t
        th.append(dict(TP=(pr & (y == 1)).sum(), FP=(pr & (y == 0)).sum(), FN=(~pr & (y == 1)).sum(), TN=(~pr & (y == 0)).sum()))
    for k in th[0]:
        avg[k] = float(np.mean([d[k] for d in th]))
    avg['Recall'] = avg['TP'] / (avg['TP'] + avg['FN']); avg['Spec'] = avg['TN'] / (avg['TN'] + avg['FP'])
    avg['Prec'] = avg['TP'] / (avg['TP'] + avg['FP'])
    avg['AUC_r'] = [float(r['AUC']) for r in rows]
    avg['hp'] = [str(h) for h in E['hp'][(0, n)]]
    avg['time'] = float(np.mean([E['time'][(r, n)] for r in range(2)]))
    T[n] = avg
# paired bootstrap Delta AUC vs GAMA (repeat 0)
g0 = E['oof'][(0, 'GAMA')]
for n in names:
    if n == 'GAMA': continue
    p0 = E['oof'][(0, n)]
    d = [roc_auc_score(y[b], g0[b]) - roc_auc_score(y[b], p0[b]) for b in boot]
    T[n]['dAUC'] = float(roc_auc_score(y, g0) - roc_auc_score(y, p0))
    T[n]['dAUC_lo'], T[n]['dAUC_hi'] = [float(v) for v in np.percentile(d, [2.5, 97.5])]
# decision curve
ts = np.arange(0.05, 0.41, 0.01)
DCA = {n: [float(np.mean([net_benefit(y, E['oof'][(r, n)], t) for r in range(2)])) for t in ts] for n in names}
DCA['Treat all'] = [float(np.mean(y) - (1 - np.mean(y)) * t / (1 - t)) for t in ts]
json.dump(dict(T=T, DCA=DCA, ts=ts.tolist(), prev=float(y.mean()), n=int(len(y)), events=int(y.sum())),
          open(RES + 'exp1_summary.json', 'w'), indent=1, default=float)
for n in names:
    print(f"{n:24s} AUC {T[n]['AUC']:.3f} ({T[n]['AUC_lo']:.3f}-{T[n]['AUC_hi']:.3f}) AP {T[n]['AP']:.3f} Brier {T[n]['Brier']:.4f} "
          f"slope {T[n]['CalSlope']:.2f} int {T[n]['CalInt']:+.2f} ECE {T[n]['ECE']:.3f} FP {T[n]['FP']:.0f} FN {T[n]['FN']:.0f} "
          + (f"dAUC {T[n]['dAUC']:+.3f} [{T[n]['dAUC_lo']:+.3f},{T[n]['dAUC_hi']:+.3f}]" if 'dAUC' in T[n] else ''))

# ---- Fig: calibration + decision curves ----
main = ['GAMA', 'LR', 'EBM', 'LightGBM']
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
for n in main:
    p = E['oof'][(0, n)]; q = np.quantile(p, np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(q, p, 'right') - 1, 0, 9)
    mp = [p[b == i].mean() for i in range(10)]; oy = [y[b == i].mean() for i in range(10)]
    ax[0].plot(mp, oy, '-o', color=COL[n], lw=2, ms=4, label=n)
ax[0].plot([0, .5], [0, .5], ':', color='#52514e', lw=1)
ax[0].set_xlabel('Mean predicted probability (decile)'); ax[0].set_ylabel('Observed event rate')
ax[0].set_title('(a) Calibration (repeat 1, pooled OOF)', fontsize=9, loc='left'); ax[0].legend(fontsize=8)
for n in main:
    ax[1].plot(ts, DCA[n], color=COL[n], lw=2, label=n)
ax[1].plot(ts, DCA['Treat all'], color='#52514e', lw=1, ls='--', label='Treat all')
ax[1].axhline(0, color='#0b0b0b', lw=0.8, label='Treat none')
ax[1].set_ylim(-0.02, 0.12); ax[1].set_xlabel('Threshold probability'); ax[1].set_ylabel('Net benefit')
ax[1].set_title('(b) Decision curves (mean of 2 repeats)', fontsize=9, loc='left'); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG + 'fig_calib_dca.png'); plt.close()

# ---- Fig: GAMA shape functions vs unconstrained GAM and LightGBM SHAP dependence ----
from collections import Counter
lam = Counter(h['lam'] for r in range(2) for h in E['hp'][(r, 'GAMA')]).most_common(1)[0][0]
lamf = Counter(h['lam'] for r in range(2) for h in E['hp'][(r, 'GAM (unconstrained)')]).most_common(1)[0][0]
g = GAMA(lam=lam).fit(X, y); gf = GAMA(lam=lamf, monotone=False).fit(X, y)
pickle.dump(dict(g=g, lam=lam), open(RES + 'gama_full.pkl', 'wb'))
import shap
lgh = Counter(str(h) for h in E['hp'][(0, 'LightGBM')]).most_common(1)[0][0]
lg = LGBM(**eval(lgh)).fit(X, y); sv = shap.TreeExplainer(lg.m).shap_values(X)
feats = ['age', 'sysBP', 'diaBP', 'totChol', 'glucose', 'BMI', 'cigsPerDay', 'heartRate']
lab = {'age': 'Age (years)', 'sysBP': 'Systolic BP (mmHg)', 'diaBP': 'Diastolic BP (mmHg)', 'totChol': 'Total cholesterol (mg/dL)',
       'glucose': 'Glucose (mg/dL)', 'BMI': 'BMI (kg/m²)', 'cigsPerDay': 'Cigarettes/day', 'heartRate': 'Heart rate (bpm)'}
fig, axs = plt.subplots(2, 4, figsize=(7.4, 4.0))
for a, f in zip(axs.flat, feats):
    lo, hi = g.k[f][0], g.k[f][-1]; grid = np.linspace(np.nanpercentile(X[f], 0.5), np.nanpercentile(X[f], 99.5), 300)
    s = g.shape(f, grid); s -= np.interp(np.nanmedian(X[f]), grid, s)
    s2 = gf.shape(f, grid); s2 -= np.interp(np.nanmedian(X[f]), grid, s2)
    xi = X[f].values; ok = ~np.isnan(xi); j = FEATS.index(f)
    a.scatter(xi[ok], sv[ok, j] - np.median(sv[ok, j]), s=1.5, color=COL['LightGBM'], alpha=0.25, lw=0, label='LightGBM TreeSHAP', rasterized=True)
    a.plot(grid, s2, color=COL['GAM (unconstrained)'], lw=1.2, ls='--', label='Unconstrained GAM')
    a.plot(grid, s, color=COL['GAMA'], lw=2, label='GAMA')
    for kv in g.k[f][1:-1]:
        a.axvline(kv, color='#52514e', lw=0.6, ls=':')
    a.set_title(lab[f], fontsize=8); a.tick_params(labelsize=7)
for a in axs[:, 0]: a.set_ylabel('Log-odds contribution', fontsize=8)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, fontsize=8, markerscale=6)
plt.tight_layout(rect=(0, 0.06, 1, 1)); plt.savefig(FIG + 'fig_shapes.png'); plt.close()
print('lam', lam, 'knots', {k: [round(float(v), 1) for v in kk] for k, kk in g.k.items()})
json.dump({'lam': lam, 'w': {f'{c}|{m}': float(w) for (c, m, _), w in zip(g.meta, g.w)}, 'b0': float(g.b0),
           'knots': {k: [float(v) for v in kk] for k, kk in g.k.items()}}, open(RES + 'gama_full.json', 'w'), indent=1)
