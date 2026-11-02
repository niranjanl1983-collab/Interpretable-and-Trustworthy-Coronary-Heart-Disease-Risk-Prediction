"""EMC: Exact Monotone Counterfactual search for GAMA (CertiRisk, Algorithm 3).

Because GAMA is additive in the logit and every interaction involves at least one
immutable feature (age, sex), the logit reduction of changing a modifiable feature j
to value v is a fixed, precomputable quantity r_j(v) given the record. For sparsity
<= 2 the minimum-cost admissible counterfactual on the action grid is found exactly:
  - singles: min cost among candidates with r_j(v) >= need
  - pairs (j,k): sort k's candidates by cost, take prefix maxima of r_k, and for each
    candidate a of j binary-search the cheapest b with prefix-max r_k >= need - r_j(a)
    (O(n log n) per pair); coupled pairs (SBP, DBP) are enumerated jointly under the
    pulse-pressure constraint.
"""
import numpy as np, time
from scipy.special import logit, expit

ACTIONS = {  # feature: (healthy floor, grid step)
    'sysBP': (110.0, 1.0), 'diaBP': (70.0, 1.0), 'totChol': (150.0, 1.0),
    'glucose': (70.0, 1.0), 'BMI': (20.0, 0.1), 'cigsPerDay': (0.0, 1.0),
}
IMMUTABLE = ['male', 'age', 'education', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']


class EMC:
    def __init__(self, model, Xtrain):
        self.m = model
        Xf = model.fill(Xtrain)
        self.scale = {}
        for f in ACTIONS:
            v = Xf[f].values
            mad = 1.4826 * np.median(np.abs(v - np.median(v)))
            self.scale[f] = mad if mad > 0 else v.std()
        pp = Xf['sysBP'] - Xf['diaBP']
        self.pp = (np.percentile(pp, 1), np.percentile(pp, 99))
        self.hi = {f: np.percentile(Xf[f], 99) for f in ACTIONS}

    def _set(self, x, f, v):
        x = x.copy(); x[f] = v
        if f == 'cigsPerDay':
            x['currentSmoker'] = 1.0 if v > 0 else 0.0
        return x

    def candidates(self, x, f):
        floor, step = ACTIONS[f]
        cur = float(x[f])
        if cur <= floor:
            return np.array([]), np.array([])
        vals = np.arange(cur - step, floor - 1e-9, -step)
        vals = np.round(vals, 6)
        cost = np.abs(cur - vals) / self.scale[f]
        return vals, cost

    def score_rows(self, rows):
        import pandas as pd
        return self.m.decision_function(pd.DataFrame(rows))

    def search(self, x, target_p, k=3):
        """x: pandas Series (one record, raw). Returns list of solutions sorted by cost."""
        import pandas as pd
        x = x.copy()
        xf = self.m.fill(pd.DataFrame([x])).iloc[0]
        z0 = self.score_rows([xf])[0]
        need = z0 - logit(target_p)
        if need <= 0:
            return [], 0
        C, Call = {}, {}
        for f in ACTIONS:
            vals, cost = self.candidates(xf, f)
            if len(vals) == 0:
                continue
            rows = [self._set(xf, f, v) for v in vals]
            red = z0 - self.score_rows(rows)
            # single-feature changes must respect pulse pressure too
            ok = np.ones(len(vals), bool)
            if f in ('sysBP', 'diaBP'):
                sb = vals if f == 'sysBP' else np.full(len(vals), xf['sysBP'])
                db = vals if f == 'diaBP' else np.full(len(vals), xf['diaBP'])
                ok = (sb - db >= self.pp[0]) & (sb - db <= self.pp[1])
            Call[f] = (vals, cost, red)            # unfiltered: used for the joint SBP-DBP pair
            C[f] = (vals[ok], cost[ok], red[ok])   # valid when the other BP value is unchanged
        sols = []
        feats = [f for f in C if len(C[f][0]) or f in ('sysBP', 'diaBP')]
        for f in feats:  # singles
            vals, cost, red = C[f]
            ok = red >= need
            if ok.any():
                i = np.argmin(np.where(ok, cost, np.inf))
                sols.append(({f: vals[i]}, cost[i]))
        for a in range(len(feats)):  # pairs
            for b in range(a + 1, len(feats)):
                fa, fb = feats[a], feats[b]
                va, ca, ra = C[fa]; vb, cb, rb = C[fb]
                if {fa, fb} == {'sysBP', 'diaBP'}:
                    # joint enumeration over unfiltered grids; terms are additive so reductions add exactly
                    va, ca, ra = Call[fa]; vb, cb, rb = Call[fb]
                    S, D = np.meshgrid(va, vb, indexing='ij')
                    cost = ca[:, None] + cb[None, :]
                    red = ra[:, None] + rb[None, :]
                    ok = (red >= need) & (S - D >= self.pp[0]) & (S - D <= self.pp[1])
                    if ok.any():
                        i, j = np.unravel_index(np.argmin(np.where(ok, cost, np.inf)), cost.shape)
                        sols.append(({fa: va[i], fb: vb[j]}, cost[i, j]))
                    continue
                o = np.argsort(cb); cbs, rbs, vbs = cb[o], rb[o], vb[o]
                pm = np.maximum.accumulate(rbs); arg = np.zeros(len(pm), int)
                best = 0
                for t in range(len(pm)):
                    if rbs[t] >= rbs[best]:
                        best = t
                    arg[t] = best
                bestc, bestsol = np.inf, None
                for i in range(len(va)):
                    res = need - ra[i]
                    if res <= 0:
                        continue  # single already covers it more cheaply
                    j = np.searchsorted(pm, res, side='left')
                    if j < len(pm):
                        c = ca[i] + cbs[arg[j]]
                        if c < bestc:
                            bestc, bestsol = c, {fa: va[i], fb: vbs[arg[j]]}
                if bestsol is not None:
                    sols.append((bestsol, bestc))
        # verify every solution by re-scoring the full model (no trust in precomputation)
        out = []
        for s, c in sorted(sols, key=lambda t: t[1]):
            xr = xf.copy()
            for f, v in s.items():
                xr = self._set(xr, f, v)
            pr = expit(self.score_rows([xr])[0])
            if pr <= target_p + 1e-12:
                out.append(dict(change=s, cost=float(c), p=float(pr)))
            if len(out) == k:
                break
        return out, len(sols)


def random_search(model_proba, x, xf, emc, target_p, budget=2000, rng=None, feats=None):
    """Baseline: random sampling of <=2-feature changes on the same admissible grid."""
    import pandas as pd
    rng = rng or np.random.default_rng(0)
    feats = feats or [f for f in ACTIONS if float(xf[f]) >= ACTIONS[f][0] + ACTIONS[f][1] - 1e-9]
    if not feats:
        return None
    rows, meta = [], []
    for _ in range(budget):
        k = rng.integers(1, min(2, len(feats)) + 1)
        fs = rng.choice(feats, size=k, replace=False)
        xr = xf.copy(); change = {}; cost = 0
        for f in fs:
            floor, step = ACTIONS[f]
            nsteps = int(np.floor((float(xf[f]) - floor) / step + 1e-9))
            v = round(float(xf[f]) - step * rng.integers(1, nsteps + 1), 6)
            xr = emc._set(xr, f, v); change[f] = v; cost += abs(float(xf[f]) - v) / emc.scale[f]
        if not (emc.pp[0] <= xr['sysBP'] - xr['diaBP'] <= emc.pp[1]):
            continue
        rows.append(xr); meta.append((change, cost))
    if not rows:
        return None
    p = model_proba(pd.DataFrame(rows))
    ok = [i for i in range(len(rows)) if p[i] <= target_p]
    if not ok:
        return None
    i = min(ok, key=lambda i: meta[i][1])
    return dict(change=meta[i][0], cost=meta[i][1], p=float(p[i]))


def greedy(model_proba, xf, emc, target_p, max_feats=2):
    """Baseline: greedy coordinate steps choosing the best risk-reduction per unit cost."""
    import pandas as pd
    xr = xf.copy(); used = []; change = {}
    p = model_proba(pd.DataFrame([xr]))[0]
    for _ in range(150):
        if p <= target_p:
            break
        best = None
        for f in ACTIONS:
            if f not in used and len(used) >= max_feats:
                continue
            floor, step = ACTIONS[f]
            st = step * (10 if f == 'BMI' else 2)
            if float(xr[f]) - st < floor - 1e-9:
                continue
            xn = emc._set(xr, f, float(xr[f]) - st)
            if not (emc.pp[0] <= xn['sysBP'] - xn['diaBP'] <= emc.pp[1]):
                continue
            pn = model_proba(pd.DataFrame([xn]))[0]
            gain = (p - pn) / (st / emc.scale[f])
            if best is None or gain > best[0]:
                best = (gain, f, xn, pn)
        if best is None or best[0] <= 0:
            return None
        _, f, xr, p = best
        if f not in used:
            used.append(f)
        change[f] = float(xr[f])
    if p > target_p:
        return None
    cost = sum(abs(float(xf[f]) - v) / emc.scale[f] for f, v in change.items())
    return dict(change=change, cost=cost, p=float(p))
