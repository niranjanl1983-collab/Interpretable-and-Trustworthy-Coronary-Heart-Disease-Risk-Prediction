"""Tables and figures for Experiments 2-5."""
import pickle, json, numpy as np, pandas as pd, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest
warnings.filterwarnings('ignore')
from analyze1_style import *

RES = '/home/claude/p2/results/'; FIG = '/home/claude/p2/figs/'
S = {}

def ci(k, n):
    if n == 0: return (np.nan, np.nan)
    r = binomtest(k, n).proportion_ci(method='exact'); return (r.low, r.high)

# ---------------- Experiment 2 ----------------
E2 = pickle.load(open(RES + 'exp2.pkl', 'rb'))
V = pd.DataFrame(E2['viol'])
vt = V.groupby(['model', 'feat'])[['v', 'vbig', 'n']].sum().reset_index()
vt['rate'] = 100 * vt.v / vt.n; vt['rate_big'] = 100 * vt.vbig / vt.n
tot = V.groupby('model')[['v', 'vbig', 'n']].sum(); tot['rate'] = 100 * tot.v / tot.n; tot['rate_big'] = 100 * tot.vbig / tot.n
S['viol'] = {m: {f: dict(rate=float(r.rate), big=float(r.rate_big), v=int(r.v), n=int(r.n)) for f, r in g.set_index('feat').iterrows()} for m, g in vt.groupby('model')}
S['viol_tot'] = {m: dict(rate=float(r.rate), big=float(r.rate_big), v=int(r.v), n=int(r.n)) for m, r in tot.iterrows()}
S['stab'] = {m: dict(rho=float(np.mean(d['rho'])), rho_sd=float(np.std(d['rho'])), rho_min=float(np.min(d['rho'])),
                     jac=float(np.mean(d['jac'])), flip=float(np.mean(d['flip']))) for m, d in E2['stab'].items()}
S['add'] = E2['add']
print(tot); print(pd.DataFrame(S['stab']).T); print(S['add'])

models = ['LR', 'EBM', 'LightGBM', 'LightGBM-mono', 'GAMA']
feats = ['age', 'sysBP', 'totChol', 'glucose', 'BMI', 'cigsPerDay']
flab = {'age': 'Age +5 y', 'sysBP': 'SBP +10', 'totChol': 'Chol +20', 'glucose': 'Glucose +10', 'BMI': 'BMI +2', 'cigsPerDay': 'Cigs +5'}
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw=dict(width_ratios=[1.6, 1]))
w = 0.16
for i, m in enumerate(models):
    vals = [S['viol'][m][f]['rate'] for f in feats]
    ax[0].bar(np.arange(len(feats)) + (i - 2) * w, vals, w * 0.9, color=COL[m], label=m)
ax[0].set_xticks(range(len(feats))); ax[0].set_xticklabels([flab[f].replace(' +', '\n+') for f in feats], fontsize=7.5)
ax[0].text(0.03, 0.55, 'GAMA and LightGBM-mono:\n0 violations', transform=ax[0].transAxes, ha='left', fontsize=7, color='#52514e')
ax[0].set_ylabel('Records with lower risk after increase (%)'); ax[0].legend(fontsize=7, ncol=2)
ax[0].set_title('(a) Local monotonicity violations (5 test folds)', fontsize=9, loc='left')
data = [E2['stab'][m]['rho'] for m in models]
bp = ax[1].boxplot(data, vert=True, patch_artist=True, widths=0.6, showfliers=False)
for patch, m in zip(bp['boxes'], models):
    patch.set_facecolor(COL[m]); patch.set_alpha(0.75); patch.set_edgecolor('#52514e')
for med in bp['medians']: med.set_color('#0b0b0b')
ax[1].set_xticklabels(['LR', 'EBM', 'LGBM', 'LGBM-\nmono', 'GAMA'], fontsize=7.5)
ax[1].set_ylabel('Spearman ρ across 10 refits'); ax[1].set_title('(b) Attribution stability', fontsize=9, loc='left')
plt.tight_layout(); plt.savefig(FIG + 'fig_expl.png'); plt.close()

# ---------------- Experiment 3-5 ----------------
E = pickle.load(open(RES + 'exp_cert.pkl', 'rb'))
# conformal
C = {}
from conformal import GROUP_NAMES
for meth in ['Mondrian (group × label)', 'Label-conditional', 'Marginal']:
    rows = [r for r in E['conf'] if r['method'] == meth]
    g = np.concatenate([r['g'] for r in rows]); yy = np.concatenate([r['y'] for r in rows]); Sx = np.concatenate([r['S'] for r in rows])
    cov = Sx[np.arange(len(yy)), yy]
    ind = Sx.all(1); emp = ~Sx.any(1)
    d = dict(cov=float(cov.mean()), indet=float(ind.mean()), empty=float(emp.mean()), n=int(len(yy)))
    single = Sx.sum(1) == 1
    d['single_acc'] = float(cov[single].mean()) if single.any() else float('nan')
    d['groups'] = {}
    for gi, gn in enumerate(GROUP_NAMES):
        mg = g == gi
        d['groups'][gn] = dict(n=int(mg.sum()), cov=float(cov[mg].mean()), cov_pos=float(cov[mg & (yy == 1)].mean()),
                               cov_neg=float(cov[mg & (yy == 0)].mean()), indet=float(ind[mg].mean()), npos=int((mg & (yy == 1)).sum()))
    d['cov_pos'] = float(cov[yy == 1].mean()); d['cov_neg'] = float(cov[yy == 0].mean())
    C[meth] = d
S['conf'] = C
for k, v in C.items(): print(k, {kk: vv for kk, vv in v.items() if kk != 'groups'}); print(pd.DataFrame(v['groups']).T.round(3))

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
meths = ['Marginal', 'Label-conditional', 'Mondrian (group × label)']
mc = ['#8a8984', '#eb6834', '#2a78d6']
for j, (key, ttl) in enumerate([('cov_pos', '(a) Coverage among CHD events'), ('cov_neg', '(b) Coverage among non-events')]):
    for i, (m, c) in enumerate(zip(meths, mc)):
        vals = [C[m]['groups'][g][key] for g in GROUP_NAMES]
        ax[j].bar(np.arange(4) + (i - 1) * 0.26, vals, 0.24, color=c, label=m)
    ax[j].axhline(0.9, color='#0b0b0b', lw=0.8, ls='--')
    ax[j].set_xticks(range(4)); ax[j].set_xticklabels(GROUP_NAMES, fontsize=7.5); ax[j].set_ylim(0, 1.05)
    ax[j].set_title(ttl, fontsize=9, loc='left')
ax[0].set_ylabel('Empirical coverage')
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, fontsize=7.5)
plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig(FIG + 'fig_conformal.png'); plt.close()

# counterfactuals
CF = E['cf']; n = len(CF)
emc_ok = [r for r in CF if r['emc']]
S['cf'] = dict(n=n, elig=int(sum(E['cf_elig'])), emc_found=len(emc_ok), rs_found=sum(1 for r in CF if r['rs']), gr_found=sum(1 for r in CF if r['gr']))
gap_rs = [r['rs']['cost'] - r['emc']['cost'] for r in emc_ok if r['rs']]
gap_gr = [r['gr']['cost'] - r['emc']['cost'] for r in emc_ok if r['gr']]
rel_rs = [(r['rs']['cost'] - r['emc']['cost']) / r['emc']['cost'] for r in emc_ok if r['rs'] and r['emc']['cost'] > 0]
rel_gr = [(r['gr']['cost'] - r['emc']['cost']) / r['emc']['cost'] for r in emc_ok if r['gr'] and r['emc']['cost'] > 0]
beaten = sum(1 for r in emc_ok for b in (r['rs'], r['gr']) if b and b['cost'] < r['emc']['cost'] - 1e-9 and len(b['change']) <= 2)
rs_only = sum(1 for r in CF if r['rs'] and not r['emc'])
S['cf'].update(beaten=beaten, rs_only=rs_only,
               emc_cost_med=float(np.median([r['emc']['cost'] for r in emc_ok])),
               rs_gap_med=float(np.median(rel_rs)), rs_gap_mean=float(np.mean(rel_rs)), rs_worse=int(sum(g > 1e-9 for g in gap_rs)), rs_n=len(gap_rs),
               gr_gap_med=float(np.median(rel_gr)), gr_gap_mean=float(np.mean(rel_gr)), gr_worse=int(sum(g > 1e-9 for g in gap_gr)), gr_n=len(gap_gr),
               t_emc=float(np.median([r['t_emc'] for r in CF])), t_rs=float(np.median([r['t_rs'] for r in CF])), t_gr=float(np.median([r['t_gr'] for r in CF])),
               spars=float(np.mean([len(r['emc']['change']) for r in emc_ok])))
from collections import Counter
use = Counter(f for r in emc_ok for f in r['emc']['change']); S['cf']['use'] = {k: v for k, v in use.items()}
S['cf']['single'] = sum(1 for r in emc_ok if len(r['emc']['change']) == 1)
noadm = [r for r in CF if not r['emc']]
S['cf']['noadm_age'] = float(np.mean([r['age'] for r in noadm])) if noadm else None
S['cf']['noadm_male'] = float(np.mean([r['male'] for r in noadm])) if noadm else None
S['cf']['adm_age'] = float(np.mean([r['age'] for r in emc_ok])); S['cf']['adm_male'] = float(np.mean([r['male'] for r in emc_ok]))
lgp = [r for r in CF if r['lg_pos']]
S['cf']['lg_n'] = len(lgp); S['cf']['lg_found'] = sum(1 for r in lgp if r['rl'])
# EMC on same record vs LightGBM random search cost is a different model; report validity only
print(S['cf'])

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
relp = [100 * v for v in rel_rs]; relg = [100 * v for v in rel_gr]
bp = ax[0].boxplot([relp, relg], vert=False, patch_artist=True, widths=0.5, showfliers=True,
                   flierprops=dict(marker='o', ms=3, mfc='#8a8984', mec='none'))
for patch, c in zip(bp['boxes'], ['#eb6834', '#1baf7a']):
    patch.set_facecolor(c); patch.set_alpha(0.75); patch.set_edgecolor('#52514e')
for med in bp['medians']: med.set_color('#0b0b0b')
ax[0].set_yticklabels([f"Random search\n(found {len(relp)}/{n})", f"Greedy descent\n(found {len(relg)}/{n})"], fontsize=7.5)
ax[0].set_xscale('symlog', linthresh=1); ax[0].set_xlabel('Excess cost over EMC optimum (%)')
ax[0].set_title(f"(a) Excess cost vs. EMC (EMC: {len(emc_ok)}/{n})", fontsize=9, loc='left')
lab = {'sysBP': 'Systolic BP', 'diaBP': 'Diastolic BP', 'totChol': 'Total chol.', 'glucose': 'Glucose', 'BMI': 'BMI', 'cigsPerDay': 'Cigarettes/day'}
ks = sorted(use, key=lambda k: -use[k])
ax[1].barh([lab[k] for k in ks][::-1], [use[k] for k in ks][::-1], color='#2a78d6', height=0.6)
ax[1].set_xlabel('Times changed in optimal counterfactual'); ax[1].set_title('(b) Features used by EMC', fontsize=9, loc='left')
plt.tight_layout(); plt.savefig(FIG + 'fig_cf.png'); plt.close()

# verifier
VR = pd.DataFrame(E['ver'])
VR['full_b'] = VR.full.apply(len) > 0; VR['bind_b'] = VR.bind.apply(len) > 0; VR['num_b'] = VR.num.apply(len) > 0
agg = VR.groupby('err')[['full_b', 'bind_b', 'num_b']].agg(['sum', 'count'])
order = ['None (valid narrative)'] + [e for e in pd.unique(VR.err) if e != 'None (valid narrative)']
S['ver'] = {e: dict(n=int(agg.loc[e, ('full_b', 'count')]), full=int(agg.loc[e, ('full_b', 'sum')]),
                    bind=int(agg.loc[e, ('bind_b', 'sum')]), num=int(agg.loc[e, ('num_b', 'sum')])) for e in order}
inj = VR[VR.err != 'None (valid narrative)']
S['ver_tot'] = dict(n=len(inj), full=int(inj.full_b.sum()), bind=int(inj.bind_b.sum()), num=int(inj.num_b.sum()),
                    valid_n=int((VR.err == 'None (valid narrative)').sum()))
for k in ('full', 'bind', 'num'):
    S['ver_tot'][k + '_ci'] = ci(S['ver_tot'][k], S['ver_tot']['n'])
S['words'] = dict(mean=float(VR.words.dropna().mean()), min=float(VR.words.dropna().min()), max=float(VR.words.dropna().max()))
reasons = Counter(r for rr in inj.full for r in rr); S['reasons'] = dict(reasons.most_common())
EQ = pd.DataFrame(E['eq'])
S['eq'] = {k: dict(n=len(g), full=int((g.full.apply(len) == 0).sum()), bind=int((g.bind.apply(len) == 0).sum()),
                   num=int((g.num.apply(len) == 0).sum())) for k, g in EQ.groupby('kind')}
print(pd.DataFrame(S['ver']).T); print(S['ver_tot']); print(S['eq'])

errs = order[1:]
M = np.array([[S['ver'][e][k] / S['ver'][e]['n'] for k in ('num', 'bind', 'full')] for e in errs])
fig, ax = plt.subplots(figsize=(5.6, 6.0))
from matplotlib.colors import LinearSegmentedColormap
cm = LinearSegmentedColormap.from_list('b', ['#f4f3ef', '#9ec3ee', '#2a78d6', '#103d73'])
ax.imshow(M, cmap=cm, vmin=0, vmax=1, aspect='auto')
for i in range(M.shape[0]):
    for j in range(3):
        ax.text(j, i, f'{100*M[i,j]:.0f}', ha='center', va='center', fontsize=7, color='white' if M[i, j] > 0.6 else '#0b0b0b')
ax.set_xticks(range(3)); ax.set_xticklabels(['B1 numeric\nmembership', 'B2 value–concept\nbinding', 'CertiRisk\ncertificate'], fontsize=8)
ax.set_yticks(range(len(errs))); ax.set_yticklabels([f"{e} (n={S['ver'][e]['n']})" for e in errs], fontsize=7.5)
ax.grid(False); ax.set_title('Injected errors detected (%)', fontsize=9, loc='left')
for s in ax.spines.values(): s.set_visible(False)
plt.tight_layout(); plt.savefig(FIG + 'fig_verifier.png'); plt.close()

json.dump(S, open(RES + 'exp2_5_summary.json', 'w'), indent=1, default=lambda o: float(o) if np.isscalar(o) else list(o))
pickle.dump(E['examples'], open(RES + 'examples.pkl', 'wb'))
ex = E['examples']
pos = [e for e in ex if 'Model counterfactual: changing' in e['text']]
neg = [e for e in ex if 'Model screen: negative' in e['text']]
json.dump(dict(pos=pos[0], neg=neg[0]), open(RES + 'example_narratives.json', 'w'), indent=1, default=float)
