"""Experiments 3-5 on repeat-0 outer folds: conformal coverage, exact counterfactuals,
and certificate verification under injected errors. Each outer training fold is split
75/25 into a proper-training set (GAMA fit) and a selection set (threshold + conformal
calibration); the outer test fold is untouched until scoring."""
import pickle, time, re, math, json
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from scipy.special import expit
from common import *
from emc import EMC, random_search, greedy, ACTIONS
from conformal import Mondrian, groups, status, GROUP_NAMES
from narrative import *

X, y = load()
E1 = pickle.load(open('/home/claude/p2/results/exp1.pkl', 'rb'))
folds = E1['folds'][0]
rng = np.random.default_rng(7)
OUT = dict(conf=[], cf=[], ver=[], eq=[], examples=[])

ERR = ['Value distortion', 'Probability distortion', 'Values swapped between concepts', 'Wrong band',
       'Direction reversal', 'Magnitude inflation (consistent odds)', 'Odds multiplier inconsistent',
       'Rank swap', 'Wrong strongest contributor', 'Top contributor omitted', 'Fabricated contributor',
       'Certificate sum broken', 'Wrong guideline category', 'Fabricated source ID', 'Invalid counterfactual',
       'Counterfactual on immutable feature', 'Causal counterfactual wording', 'Conformal status flip',
       'Treatment order added', 'Fasting assumption', 'Screen reversal']


def lines(t): return t.split('\n')
def idx(L, prefix): return next((i for i, l in enumerate(L) if l.startswith(prefix)), None)


def inject(kind, t, P, model):
    L = lines(t)
    c1, c2, c3 = idx(L, 'Contributor 1:'), idx(L, 'Contributor 2:'), idx(L, 'Contributor 3:')
    if kind == 'Value distortion':
        i = idx(L, 'Blood pressure'); s = P['x']['sysBP']
        L[i] = L[i].replace(f'Blood pressure {fmt(s)}/', f'Blood pressure {fmt(s + 1)}/', 1)
    elif kind == 'Probability distortion':
        i = idx(L, 'Ten-year'); L[i] = L[i].replace(f'{100*P["p"]:.1f}%', f'{100*P["p"]+3:.1f}%', 1)
    elif kind == 'Values swapped between concepts':
        i, j = idx(L, 'Total cholesterol'), idx(L, 'Glucose')
        a, b = fmt(P['x']['totChol']), fmt(P['x']['glucose'])
        if a == b: return None
        L[i] = L[i].replace(f'cholesterol {a} ', f'cholesterol {b} ', 1); L[j] = L[j].replace(f'Glucose {b} ', f'Glucose {a} ', 1)
    elif kind == 'Wrong band':
        cand = [k for k in (c1, c2, c3) if k is not None and '(band ' in L[k]]
        if not cand: return None
        k = cand[0]; f = re.search(r'\[GAMA:(\w+)\]', L[k]).group(1); kn = model.k[f]
        m = model.band(f, P['x'][f]); alt = kn[m] + 0.01 if m < len(kn) - 1 else kn[m - 1] - 0.01
        old = band_text(model, f, P['x'][f]); new = band_text(model, f, alt)
        if old == new: return None
        L[k] = L[k].replace(f'(band {old})', f'(band {new})')
    elif kind == 'Direction reversal':
        L[c1] = L[c1].replace(' raised ', ' §lowered ').replace(' lowered ', ' raised ').replace('§', '') if ' raised ' in L[c1] else L[c1].replace(' lowered ', ' raised ')
    elif kind == 'Magnitude inflation (consistent odds)':
        c = P['contrib'][P['top'][0]]; nm = abs(c) + 0.30; sign = 1 if c >= 0 else -1
        L[c1] = re.sub(r'by [\d.]+ \(odds ×[\d.]+\)', f'by {nm:.2f} (odds ×{math.exp(sign*nm):.2f})', L[c1])
    elif kind == 'Odds multiplier inconsistent':
        c = P['contrib'][P['top'][0]]
        L[c1] = re.sub(r'\(odds ×[\d.]+\)', f'(odds ×{math.exp(c)+0.5:.2f})', L[c1])
    elif kind == 'Rank swap':
        L[c1], L[c2] = L[c2].replace('Contributor 2:', 'Contributor 1:'), L[c1].replace('Contributor 1:', 'Contributor 2:')
    elif kind == 'Wrong strongest contributor':
        i = idx(L, 'Strongest'); L[i] = f'Strongest contributor: {NAME[P["top"][1]]}.'
    elif kind == 'Top contributor omitted':
        L[c2] = L[c2].replace('Contributor 2:', 'Contributor 1:'); L[c3] = L[c3].replace('Contributor 3:', 'Contributor 2:')
        del L[c1]
    elif kind == 'Fabricated contributor':
        cand = [f for f in ['heartRate', 'BMI', 'totChol', 'glucose', 'diaBP', 'education'] if f not in P['top'] and pd.notna(P['x'][f])]
        f = min(cand, key=lambda f: abs(P['contrib'][f]))
        L[c3] = (f"Contributor 3: {NAME[f]} = {fmt(P['x'][f], f)} {UNIT[f]} (band {band_text(model, f, P['x'][f])}) "
                 f"raised the log-odds by 0.25 (odds ×{math.exp(0.25):.2f}). [GAMA:{f}]")
    elif kind == 'Certificate sum broken':
        i = idx(L, 'Certificate'); L[i] = L[i].replace(f'baseline log-odds {P["base"]:+.2f}', f'baseline log-odds {P["base"]+0.2:+.2f}')
    elif kind == 'Wrong guideline category':
        i = idx(L, 'Total cholesterol'); cur = chol_atp3(P['x']['totChol'])
        alt = 'desirable' if cur != 'desirable' else 'high'
        L[i] = L[i].replace(f': {cur} [', f': {alt} [')
    elif kind == 'Fabricated source ID':
        i = idx(L, 'Blood pressure'); L[i] = L[i].replace('[ACC_AHA_BP_2017]', '[ACC_AHA_BP_2099]')
    elif kind == 'Invalid counterfactual':
        i = idx(L, 'Model counterfactual: changing')
        if i is None: return None
        f, v = list(P['cf']['change'].items())[0]; x0 = P['x'][f]
        half = round((x0 + v) / 2) if f != 'BMI' else round((x0 + v) / 2, 1)
        if half == v: return None
        L[i] = L[i].replace(f'from {fmt(x0)} to {fmt(v)}', f'from {fmt(x0)} to {fmt(half)}', 1)
    elif kind == 'Counterfactual on immutable feature':
        i = idx(L, 'Model counterfactual: changing')
        if i is None: return None
        a = P['x']['age']
        L[i] = re.sub(r'changing .+? would move', f'changing age from {fmt(a)} to {fmt(a-10)} would move', L[i])
    elif kind == 'Causal counterfactual wording':
        i = idx(L, 'Model counterfactual: changing')
        if i is None: return None
        L[i] = L[i].replace('would move the model probability to', "will reduce the patient's risk to").replace(' ' + CF_DISCLAIMER, '')
    elif kind == 'Conformal status flip':
        i = idx(L, 'Conformal status'); cur = re.search(r': (\w+)\.$', L[i]).group(1)
        L[i] = L[i].replace(f': {cur}.', ': ' + ('positive' if cur != 'positive' else 'negative') + '.')
    elif kind == 'Treatment order added':
        L.insert(len(L) - 1, 'Start statin therapy and initiate antihypertensive medication.')
    elif kind == 'Fasting assumption':
        i = idx(L, 'Glucose'); L[i] = L[i].replace('fasting status unverified, no glycaemic category assigned', 'fasting confirmed, diabetes range')
    elif kind == 'Screen reversal':
        i = idx(L, 'Model screen'); L[i] = L[i].replace('positive', '§').replace('negative', 'positive').replace('§', 'negative')
    out = '\n'.join(L)
    return None if out == t else out


def equivalents(t, P):
    L = lines(t); E = {}
    i, j = idx(L, 'Blood pressure'), idx(L, 'Body-mass')
    L2 = L.copy(); L2[i], L2[j] = L[j], L[i]; E['Guideline sentences reordered'] = '\n'.join(L2)
    L2 = L.copy(); k = idx(L, 'Ten-year')
    L2[k] = f"Ten-year CHD model probability: {P['p']:.3f} (research threshold {P['thr']:.3f})."
    E['Probability as a fraction'] = '\n'.join(L2)
    E['Direction synonyms'] = t.replace(' raised the log-odds', ' increased the log-odds').replace(' lowered the log-odds', ' decreased the log-odds')
    E['Odds phrasing'] = t.replace('(odds ×', '(odds multiplied by ')
    return E


for k, (tr, te) in enumerate(folds):
    tr, te = np.array(tr), np.array(te)
    lam = E1['hp'][(0, 'GAMA')][k]['lam']
    trp, sel = train_test_split(tr, test_size=0.25, stratify=y[tr], random_state=300 + k)
    m = GAMA(lam=lam).fit(X.iloc[trp], y[trp])
    psel = m.predict_proba(X.iloc[sel])[:, 1]; thr = threshold_for_recall(y[sel], psel, 0.75)
    pte = m.predict_proba(X.iloc[te])[:, 1]; gte = groups(X.iloc[te])
    # ---- conformal ----
    for name, kw in [('Mondrian (group × label)', dict(by_group=True, by_label=True)),
                     ('Label-conditional', dict(by_group=False, by_label=True)),
                     ('Marginal', dict(by_group=False, by_label=False))]:
        cp = Mondrian(0.10, **kw).fit(psel, y[sel], groups(X.iloc[sel]))
        S = cp.predict_sets(pte, gte)
        OUT['conf'].append(dict(fold=k, method=name, g=gte, y=y[te], S=S))
    cpM = Mondrian(0.10).fit(psel, y[sel], groups(X.iloc[sel]))
    stat = status(cpM.predict_sets(pte, gte))
    # ---- counterfactuals (complete-case screened-positive test records) ----
    emc = EMC(m, X.iloc[trp]); P_g = lambda D: m.predict_proba(D)[:, 1]
    lg_hp = E1['hp'][(0, 'LightGBM')][k]; lg = LGBM(**lg_hp).fit(X.iloc[trp], y[trp])
    P_l = lambda D: lg.predict_proba(D)[:, 1]
    lsel = P_l(X.iloc[sel]); lthr = threshold_for_recall(y[sel], lsel, 0.75)
    target = thr - 0.001
    cfs = {}
    elig = [ii for ii, i in enumerate(te) if not X.iloc[i].isna().any() and pte[ii] >= thr]
    OUT.setdefault('cf_elig', []).append(len(elig))
    cf_pick = set(rng.choice(elig, min(60, len(elig)), replace=False).tolist())
    for ii, i in enumerate(te):
        xr = X.iloc[i]
        if ii not in cf_pick:
            continue
        t0 = time.time(); sols, nsol = emc.search(xr, target); t_emc = time.time() - t0
        xf = m.fill(X.iloc[[i]]).iloc[0]
        t0 = time.time(); rs = random_search(P_g, xr, xf, emc, target, budget=2000, rng=np.random.default_rng(i)); t_rs = time.time() - t0
        t0 = time.time(); gr = greedy(P_g, xf, emc, target); t_gr = time.time() - t0
        rl = None
        if P_l(X.iloc[[i]])[0] >= lthr:
            t0 = time.time(); rl = random_search(P_l, xr, xf, emc, lthr - 0.001, budget=2000, rng=np.random.default_rng(i)); t_rl = time.time() - t0
        else:
            t_rl = None
        OUT['cf'].append(dict(fold=k, i=int(i), p=float(pte[ii]), emc=sols[0] if sols else None, emc_all=sols,
                              t_emc=t_emc, rs=rs, t_rs=t_rs, gr=gr, t_gr=t_gr, lg_pos=t_rl is not None, rl=rl, t_rl=t_rl,
                              age=float(xr['age']), male=int(xr['male'])))
        cfs[i] = sols[0] if sols else None
    # ---- narratives + verification ----
    cc = [ii for ii, i in enumerate(te) if not X.iloc[i].isna().any()]
    pos = [ii for ii in cc if pte[ii] >= thr]; neg = [ii for ii in cc if pte[ii] < thr]
    pick = list(rng.choice(sorted(cf_pick), 25, replace=False)) + list(rng.choice(neg, 15, replace=False))
    for ii in pick:
        i = te[ii]; x = X.iloc[i].to_dict()
        cf = cfs.get(i)
        P = payload(m, x, f'F{k+1}_R{i:04d}', thr, conformal=stat[ii], cf=cf)
        t = compile_narrative(m, P)
        kw = dict(conformal=stat[ii])
        base = (verify_full(t, m, x, thr, **kw), verify_binding(t, m, x, thr), verify_numeric(t, m, x, thr, cf=cf))
        OUT['ver'].append(dict(fold=k, i=int(i), err='None (valid narrative)', full=base[0], bind=base[1], num=base[2],
                               words=len(t.split())))
        if len(OUT['examples']) < 40:
            OUT['examples'].append(dict(i=int(i), text=t, p=P['p'], thr=thr, y=int(y[i]), status=stat[ii]))
        for e in ERR:
            tt = inject(e, t, P, m)
            if tt is None:
                continue
            OUT['ver'].append(dict(fold=k, i=int(i), err=e, full=verify_full(tt, m, x, thr, **kw),
                                   bind=verify_binding(tt, m, x, thr), num=verify_numeric(tt, m, x, thr, cf=cf), text=tt))
        for e, tt in equivalents(t, P).items():
            OUT['eq'].append(dict(fold=k, i=int(i), kind=e, full=verify_full(tt, m, x, thr, **kw),
                                  bind=verify_binding(tt, m, x, thr), num=verify_numeric(tt, m, x, thr, cf=cf)))
    print('fold', k, 'done', len(OUT['cf']), len(OUT['ver']), flush=True)
pickle.dump(OUT, open('/home/claude/p2/results/exp_cert.pkl', 'wb'))
