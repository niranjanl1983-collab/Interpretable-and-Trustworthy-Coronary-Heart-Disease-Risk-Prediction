"""Certified Narrative Compiler (CNC) and Certificate Verifier (CertiRisk, Algorithm 2).

The compiler turns a GAMA prediction into a narrative whose every sentence carries a
certificate (term, value, unit, band, contribution, odds multiplier, rank, rule id).
The verifier is given only the narrative text, the record and the frozen model; it
re-derives every certified quantity and accepts the narrative only if all checks pass.
Two weaker verifiers are provided as comparators:
  B1 numeric membership   - every number in the text occurs in the record/payload
  B2 value-concept binding - values, units, categories, sources, contributor set,
                             directions, screen and scope (the style of earlier work)
"""
import re, math
import numpy as np, pandas as pd
from scipy.special import expit, logit

NAME = {'age': 'age', 'sysBP': 'systolic blood pressure', 'diaBP': 'diastolic blood pressure',
        'totChol': 'total cholesterol', 'glucose': 'glucose', 'BMI': 'body-mass index',
        'heartRate': 'heart rate', 'cigsPerDay': 'smoking intensity', 'education': 'education level',
        'male': 'male sex', 'currentSmoker': 'current smoking', 'BPMeds': 'blood-pressure medication',
        'prevalentStroke': 'prior stroke', 'prevalentHyp': 'prevalent hypertension', 'diabetes': 'diabetes'}
INV = {v: k for k, v in NAME.items()}
UNIT = {'age': 'years', 'sysBP': 'mmHg', 'diaBP': 'mmHg', 'totChol': 'mg/dL', 'glucose': 'mg/dL',
        'BMI': 'kg/m²', 'heartRate': 'bpm', 'cigsPerDay': 'cigarettes/day', 'education': 'level'}
BINARY = ['male', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']
MODIFIABLE = ['sysBP', 'diaBP', 'totChol', 'glucose', 'BMI', 'cigsPerDay']
SOURCES = {'ACC_AHA_BP_2017', 'JNC7_2003', 'ATP_III_2002', 'ADA_2024', 'WHO_BMI_2000'}
SCOPE = 'Scope: research decision-support output; not a diagnosis; requires clinician review.'
CF_DISCLAIMER = 'This describes the model, not the effect of treatment.'
FORBIDDEN = [r'\bdiagnos(is|ed) (of|with|is confirmed)', r'\bconfirmed diagnosis\b', r'\bstart\b', r'\bprescribe',
             r'\binitiate\b', r'\bwill (reduce|lower|prevent)', r'\bfasting (status )?(is )?confirmed\b',
             r'\bdiabetes range\b', r'\bcaus(es|ed|al effect)\b']


# ---------------- guideline registry ----------------
def bp_acc2017(s, d):
    if s >= 140 or d >= 90: return 'stage 2'
    if s >= 130 or d >= 80: return 'stage 1'
    if s >= 120: return 'elevated'
    return 'normal'


def bp_jnc7(s, d):
    if s >= 160 or d >= 100: return 'stage 2'
    if s >= 140 or d >= 90: return 'stage 1'
    if s >= 120 or d >= 80: return 'prehypertension'
    return 'normal'


def chol_atp3(c):
    return 'high' if c >= 240 else ('borderline high' if c >= 200 else 'desirable')


def bmi_who(b):
    if b < 18.5: return 'underweight'
    if b < 25: return 'normal weight'
    if b < 30: return 'overweight'
    return 'obesity'


def fmt(v, f=None):
    if f in BINARY:
        return 'yes' if v >= 0.5 else 'no'
    if f == 'BMI':
        return f'{v:.1f}'
    return f'{v:g}' if abs(v - round(v, 1)) < 1e-9 else f'{v:.2f}'


def band_text(model, f, v):
    kn = model.k[f]; m = model.band(f, v); u = UNIT[f]
    inner = kn[1:-1]
    if not inner:
        return f'full range {u}'
    if m == 1:
        return f'below {inner[0]:g} {u}'
    if m == len(kn) - 1:
        return f'{inner[-1]:g} {u} and above'
    return f'{kn[m-1]:g}–{kn[m]:g} {u}'


# ---------------- compiler ----------------
def payload(model, x_raw, rid, thr, conformal=None, cf=None, k=3):
    X1 = pd.DataFrame([x_raw])
    base, C, names = model.contributions(X1)
    C = C[0]
    z = base + C.sum()
    main = [n for n in names if n in NAME]
    order = sorted(main, key=lambda n: -abs(C[names.index(n)]))
    top = order[:k]
    listed = sum(C[names.index(n)] for n in top)
    return dict(rid=rid, x=x_raw, p=float(expit(z)), z=float(z), base=float(base), thr=thr,
                contrib={n: float(C[names.index(n)]) for n in names}, top=top,
                listed=float(listed), other=float(z - base - listed), conformal=conformal, cf=cf)


def compile_narrative(model, P):
    x = P['x']; L = []
    L.append(f"Narrative for record {P['rid']}.")
    L.append(f"Ten-year CHD model probability: {100*P['p']:.1f}% (research threshold {100*P['thr']:.1f}%).")
    L.append(f"Model screen: {'positive' if P['p'] >= P['thr'] else 'negative'}.")
    if P['conformal'] is not None:
        L.append(f"Conformal status at 90% confidence: {P['conformal']}.")
    L.append(f"Certificate: baseline log-odds {P['base']:+.2f}; listed contributions {P['listed']:+.2f}; "
             f"other terms {P['other']:+.2f}; total log-odds {P['z']:+.2f}.")
    for i, f in enumerate(P['top'], 1):
        c = P['contrib'][f]
        verb = 'raised' if c >= 0 else 'lowered'
        if f in BINARY:
            val = f"{NAME[f]} = {fmt(x[f], f)}"
        else:
            val = f"{NAME[f]} = {fmt(x[f], f)} {UNIT[f]} (band {band_text(model, f, x[f])})"
        L.append(f"Contributor {i}: {val} {verb} the log-odds by {abs(c):.2f} (odds ×{math.exp(c):.2f}). [GAMA:{f}]")
    L.append(f"Strongest contributor: {NAME[P['top'][0]]}.")
    s, d = x['sysBP'], x['diaBP']
    L.append(f"Blood pressure {fmt(s)}/{fmt(d)} mmHg: {bp_acc2017(s, d)} [ACC_AHA_BP_2017]; {bp_jnc7(s, d)} [JNC7_2003].")
    L.append(f"Total cholesterol {fmt(x['totChol'])} mg/dL: {chol_atp3(x['totChol'])} [ATP_III_2002].")
    L.append(f"Glucose {fmt(x['glucose'])} mg/dL: fasting status unverified, no glycaemic category assigned [ADA_2024].")
    L.append(f"Body-mass index {fmt(x['BMI'], 'BMI')} kg/m²: {bmi_who(x['BMI'])} [WHO_BMI_2000].")
    if P['p'] >= P['thr']:
        if P['cf']:
            parts = [f"{NAME[f]} from {fmt(x[f])} to {fmt(v)}" for f, v in P['cf']['change'].items()]
            L.append(f"Model counterfactual: changing {' and '.join(parts)} would move the model probability to "
                     f"{100*P['cf']['p']:.1f}%, below the threshold. {CF_DISCLAIMER}")
        else:
            L.append("Model counterfactual: no admissible change of at most two modifiable measurements brings "
                     "the probability below the threshold.")
    L.append(SCOPE)
    return '\n'.join(L)


# ---------------- parsing helpers ----------------
NUM = r'[-+]?\d+(?:\.\d+)?'


def _num(s):
    return float(s)


def parse(text):
    """Parse a narrative into typed claims. Unrecognised lines are returned separately."""
    out = dict(contrib=[], unknown=[], guide={}, raw=text)
    for line in [l.strip() for l in text.split('\n') if l.strip()]:
        m = re.match(rf'^Narrative for record (\S+)\.$', line)
        if m: out['rid'] = m.group(1); continue
        m = re.match(rf'^Ten-year CHD model probability: ({NUM})(%?) \(research threshold ({NUM})(%?)\)\.$', line)
        if m:
            pv = _num(m.group(1)) / (100 if m.group(2) else 1)
            tv = _num(m.group(3)) / (100 if m.group(4) else 1)
            out['p'] = pv; out['p_dec'] = len(m.group(1).split('.')[-1]) if '.' in m.group(1) else 0
            out['p_pct'] = bool(m.group(2)); out['thr'] = tv; continue
        m = re.match(r'^Model screen: (positive|negative)\.$', line)
        if m: out['screen'] = m.group(1); continue
        m = re.match(r'^Conformal status at 90% confidence: (positive|negative|indeterminate)\.$', line)
        if m: out['conformal'] = m.group(1); continue
        m = re.match(rf'^Certificate: baseline log-odds ({NUM}); listed contributions ({NUM}); other terms ({NUM}); total log-odds ({NUM})\.$', line)
        if m: out['cert'] = tuple(_num(g) for g in m.groups()); continue
        m = re.match(rf'^Contributor (\d+): (.+?) = (yes|no|{NUM})(?: (\S+) \(band (.+?)\))? '
                     rf'(raised|lowered|increased|decreased) the log-odds by ({NUM}) \((?:odds ×|odds multiplied by )({NUM})\)\. \[GAMA:(\w+)\]$', line)
        if m:
            out['contrib'].append(dict(rank=int(m.group(1)), name=m.group(2), val=m.group(3), unit=m.group(4),
                                       band=m.group(5), up=m.group(6) in ('raised', 'increased'),
                                       mag=_num(m.group(7)), odds=_num(m.group(8)), tag=m.group(9)))
            continue
        m = re.match(r'^Strongest contributor: (.+)\.$', line)
        if m: out['strongest'] = m.group(1); continue
        m = re.match(rf'^Blood pressure ({NUM})/({NUM}) mmHg: (.+?) \[(\w+)\]; (.+?) \[(\w+)\]\.$', line)
        if m:
            out['guide']['bp'] = (_num(m.group(1)), _num(m.group(2)), m.group(3), m.group(4), m.group(5), m.group(6)); continue
        m = re.match(rf'^Total cholesterol ({NUM}) mg/dL: (.+?) \[(\w+)\]\.$', line)
        if m: out['guide']['chol'] = (_num(m.group(1)), m.group(2), m.group(3)); continue
        m = re.match(rf'^Glucose ({NUM}) mg/dL: (.+?) \[(\w+)\]\.$', line)
        if m: out['guide']['glu'] = (_num(m.group(1)), m.group(2), m.group(3)); continue
        m = re.match(rf'^Body-mass index ({NUM}) kg/m²: (.+?) \[(\w+)\]\.$', line)
        if m: out['guide']['bmi'] = (_num(m.group(1)), m.group(2), m.group(3)); continue
        m = re.match(rf'^Model counterfactual: changing (.+) would move the model probability to ({NUM})%, below the threshold\. (.*)$', line)
        if m:
            ch = {}
            for part in m.group(1).split(' and '):
                mm = re.match(rf'^(.+?) from ({NUM}) to ({NUM})$', part)
                if mm: ch[mm.group(1)] = (_num(mm.group(2)), _num(mm.group(3)))
                else: ch[part] = None
            out['cf'] = dict(change=ch, p=_num(m.group(2)) / 100, tail=m.group(3)); continue
        if line.startswith('Model counterfactual: no admissible change'):
            out['cf'] = 'none'; continue
        if line == SCOPE:
            out['scope'] = True; continue
        out['unknown'].append(line)
    return out


# ---------------- verifiers ----------------
def _val_ok(stated, true, f=None):
    if f in BINARY:
        return stated == ('yes' if true >= 0.5 else 'no')
    try:
        return abs(float(stated) - float(true)) <= (0.051 if f == 'BMI' else 0.0051)
    except ValueError:
        return False


def _forbidden(text):
    t = text.lower().replace(CF_DISCLAIMER.lower(), '')
    return [p for p in FORBIDDEN if re.search(p, t)]


def verify_full(text, model, x, thr, conformal=None, emc_target=None, k=3):
    """CertiRisk certificate verifier. Returns list of failure reasons (empty = pass)."""
    R = []; Q = parse(text)
    if Q['unknown']: R.append('unrecognised_prose')
    P = payload(model, x, Q.get('rid', '?'), thr)
    # probability and threshold
    if 'p' not in Q: R.append('missing_probability')
    else:
        tol = 0.5 * 10 ** (-Q['p_dec']) / (100 if Q['p_pct'] else 1) + 1e-9
        if abs(Q['p'] - P['p']) > tol: R.append('probability')
        if abs(Q['thr'] - thr) > 0.0006: R.append('threshold')
    scr = 'positive' if P['p'] >= thr else 'negative'
    if Q.get('screen') != scr: R.append('screen')
    if conformal is not None and Q.get('conformal') != conformal: R.append('conformal')
    # certificate integrity: recompute every component from the frozen model
    if 'cert' not in Q: R.append('missing_certificate')
    else:
        b, l, o, z = Q['cert']
        if abs(b - P['base']) > 0.0051: R.append('cert_base')
        if abs(z - P['z']) > 0.0051: R.append('cert_total')
        if abs(b + l + o - z) > 0.0151: R.append('cert_sum')
        if abs(expit(z) - P['p']) > 0.002: R.append('cert_prob')
        if abs(l - P['listed']) > 0.0051 or abs(o - P['other']) > 0.0051: R.append('cert_partition')
    # contributors: identity, values, bands, directions, magnitudes, odds, ranks
    names = [c['name'] for c in Q['contrib']]
    feats = [INV.get(n) for n in names]
    if feats != P['top']:
        if set(feats) != set(P['top']): R.append('contributor_set')
        else: R.append('contributor_rank')
    for c in Q['contrib']:
        f = INV.get(c['name'])
        if f is None or c['tag'] != f: R.append('contributor_identity'); continue
        if not _val_ok(c['val'], x[f], f): R.append('contributor_value')
        if f not in BINARY:
            if c['unit'] != UNIT[f]: R.append('contributor_unit')
            if c['band'] != band_text(model, f, x[f]): R.append('contributor_band')
        tc = P['contrib'][f]
        if c['up'] != (tc >= 0): R.append('direction')
        if abs(c['mag'] - abs(tc)) > 0.0051: R.append('magnitude')
        if abs(c['odds'] - math.exp(tc)) > 0.0051 + 0.01 * math.exp(tc): R.append('odds')
        if c['rank'] != (P['top'].index(f) + 1 if f in P['top'] else -1): R.append('contributor_rank')
    if Q.get('strongest') != NAME[P['top'][0]]: R.append('strongest')
    R += _guides(Q, x)
    # counterfactual: re-score with the frozen model; only modifiable features may change
    if P['p'] >= thr:
        if 'cf' not in Q: R.append('cf_missing')
        elif Q['cf'] == 'none':
            if emc_target is not None and emc_target: R.append('cf_omitted')
        else:
            xr = dict(x); ok = True
            for nm, ch in Q['cf']['change'].items():
                f = INV.get(nm)
                if f is None or ch is None: R.append('cf_parse'); ok = False; continue
                if f not in MODIFIABLE: R.append('cf_immutable'); ok = False; continue
                if abs(ch[0] - x[f]) > 0.051: R.append('cf_from_value')
                xr[f] = ch[1]
                if f == 'cigsPerDay': xr['currentSmoker'] = 1.0 if ch[1] > 0 else 0.0
            if len(Q['cf']['change']) > 2: R.append('cf_sparsity')
            if ok:
                pr = float(model.predict_proba(pd.DataFrame([xr]))[0, 1])
                if abs(pr - Q['cf']['p']) > 0.0006: R.append('cf_probability')
                if pr >= thr: R.append('cf_invalid')
            if Q['cf']['tail'] != CF_DISCLAIMER: R.append('cf_disclaimer')
    elif 'cf' in Q: R.append('cf_unexpected')
    if not Q.get('scope'): R.append('scope')
    if _forbidden(text): R.append('forbidden_claim')
    return sorted(set(R))


def _guides(Q, x):
    R = []; G = Q['guide']
    if 'bp' not in G: R.append('guide_missing')
    else:
        s, d, c1, s1, c2, s2 = G['bp']
        if abs(s - x['sysBP']) > 0.0051 or abs(d - x['diaBP']) > 0.0051: R.append('guide_value')
        if (c1, s1) != (bp_acc2017(x['sysBP'], x['diaBP']), 'ACC_AHA_BP_2017'): R.append('guide_category' if s1 in SOURCES else 'source')
        if (c2, s2) != (bp_jnc7(x['sysBP'], x['diaBP']), 'JNC7_2003'): R.append('guide_category' if s2 in SOURCES else 'source')
    for key, f, fn, src in [('chol', 'totChol', chol_atp3, 'ATP_III_2002'), ('bmi', 'BMI', bmi_who, 'WHO_BMI_2000')]:
        if key not in G: R.append('guide_missing'); continue
        v, cat, s = G[key]
        if not _val_ok(v, x[f], f): R.append('guide_value')
        if s != src: R.append('source')
        elif cat != fn(x[f]): R.append('guide_category')
    if 'glu' not in G: R.append('guide_missing')
    else:
        v, cat, s = G['glu']
        if abs(v - x['glucose']) > 0.0051: R.append('guide_value')
        if s != 'ADA_2024': R.append('source')
        if cat != 'fasting status unverified, no glycaemic category assigned': R.append('guide_category')
    return R


def verify_binding(text, model, x, thr, **kw):
    """B2: value-concept binding in the style of earlier work (no certificate arithmetic,
    no magnitudes/odds/ranks, no counterfactual re-scoring, no conformal check)."""
    R = []; Q = parse(text)
    if Q['unknown']: R.append('unrecognised_prose')
    P = payload(model, x, '?', thr)
    if 'p' not in Q or abs(Q['p'] - P['p']) > 0.0006: R.append('probability')
    if Q.get('screen') != ('positive' if P['p'] >= thr else 'negative'): R.append('screen')
    feats = [INV.get(c['name']) for c in Q['contrib']]
    if set(feats) != set(P['top']): R.append('contributor_set')
    for c in Q['contrib']:
        f = INV.get(c['name'])
        if f is None: R.append('contributor_identity'); continue
        if not _val_ok(c['val'], x[f], f): R.append('contributor_value')
        if f not in BINARY and c['unit'] != UNIT[f]: R.append('contributor_unit')
        if c['up'] != (P['contrib'][f] >= 0): R.append('direction')
    R += _guides(Q, x)
    if not Q.get('scope'): R.append('scope')
    if _forbidden(text): R.append('forbidden_claim')
    return sorted(set(R))


def verify_numeric(text, model, x, thr, cf=None, **kw):
    """B1: every number in the narrative must appear in the record or payload."""
    P = payload(model, x, '?', thr, cf=cf)
    allowed = set()
    def add(v):
        for d in (0, 1, 2):
            allowed.add(round(float(v), d))
    for f, v in x.items():
        if pd.notna(v): add(v)
    for v in (100 * P['p'], 100 * thr, P['base'], P['listed'], P['other'], P['z']):
        add(v); add(abs(v))
    for f, c in P['contrib'].items():
        add(c); add(abs(c)); add(math.exp(c))
    for f in model.k:
        for kv in model.k[f]: add(kv)
    if cf:
        add(100 * cf['p'])
        for v in cf['change'].values(): add(v)
    for v in (90, 10, 1, 2, 3, 2017, 2003, 2002, 2024, 2000, 7, 8):
        add(v)
    body = re.sub(r'\[[A-Z_a-z0-9:]+\]', '', text)
    body = re.sub(r'record \S+\.', '', body)
    nums = [float(s) for s in re.findall(r'\d+(?:\.\d+)?', body)]
    bad = [n for n in nums if round(n, 2) not in allowed and round(n, 1) not in allowed and round(n) not in allowed]
    R = ['numeric_membership'] if bad else []
    if len(text.split()) > 450: R.append('length')
    return R
