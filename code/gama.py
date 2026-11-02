"""GAMA: Guideline-Anchored Monotone Additive risk model (CertiRisk, Algorithm 1).

logit p(x) = b0 + sum_j g_j(x_j) + sum_(j,k) h_jk
Continuous features use a normalised ramp basis between anchor knots:
    r_m(x) = clip((x - k_{m-1}) / (k_m - k_{m-1}), 0, 1),  m = 1..M
so g_j(x) = sum_m w_jm r_m(x) and w_jm is the logit rise across band m.
Monotone non-decreasing  <=>  w_jm >= 0 (box constraint, L-BFGS-B).
Values outside [k_0, k_M] (training 1st/99th percentiles) are clipped,
so extrapolation is flat and every contribution is bounded.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

CONT = ['age', 'sysBP', 'diaBP', 'totChol', 'glucose', 'BMI', 'heartRate', 'cigsPerDay', 'education']
BIN = ['male', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']
MISS = ['glucose', 'totChol', 'cigsPerDay', 'BPMeds', 'education']

# Guideline / clinical anchor knots (interior). Sources in manuscript Table III.
ANCHORS = {
    'sysBP': [120, 130, 140, 160],      # ACC/AHA 2017 + JNC 7 stage 2
    'diaBP': [80, 90, 100],             # ACC/AHA 2017 stage 1/2, JNC 7 stage 2
    'totChol': [200, 240],              # ATP III
    'glucose': [100, 126],              # ADA (fasting) cut-points used as knots only
    'BMI': [18.5, 25, 30, 35],          # WHO adult BMI classes
    'age': [45, 55, 65],                # risk-band anchors
    'heartRate': [60, 100],             # brady/tachycardia definitions
    'cigsPerDay': [1, 10, 20],          # never / light / moderate / heavy
    'education': [2, 3],
}
# Prespecified clinical direction priors: +1 non-decreasing, 0 free.
MONO = {'age': 1, 'sysBP': 1, 'diaBP': 0, 'totChol': 1, 'glucose': 1, 'BMI': 1,
        'heartRate': 0, 'cigsPerDay': 1, 'education': 0,
        'male': 1, 'currentSmoker': 1, 'BPMeds': 0, 'prevalentStroke': 1,
        'prevalentHyp': 1, 'diabetes': 1}
INTER = [('age', 'sysBP'), ('male', 'age')]

UNITS = {'age': 'years', 'sysBP': 'mmHg', 'diaBP': 'mmHg', 'totChol': 'mg/dL', 'glucose': 'mg/dL',
         'BMI': 'kg/m²', 'heartRate': 'bpm', 'cigsPerDay': 'cigarettes/day', 'education': 'level'}


class GAMA:
    def __init__(self, lam=1.0, knots='anchor', monotone=True, interactions=False, n_q=None):
        self.lam, self.knots, self.monotone, self.interactions = lam, knots, monotone, interactions

    # ---------- preprocessing (fitted on training data only) ----------
    def _prep_fit(self, X):
        self.med = {c: float(np.nanmedian(X[c])) for c in CONT + BIN}
        self.k = {}
        for c in CONT:
            v = X[c].dropna().values
            lo, hi = np.percentile(v, 1), np.percentile(v, 99)
            if self.knots == 'anchor':
                inner = [a for a in ANCHORS[c] if lo < a < hi]
            else:  # quantile knots, same count as anchors (ablation)
                n = len([a for a in ANCHORS[c] if lo < a < hi]) or 1
                inner = list(np.unique(np.percentile(v, np.linspace(0, 100, n + 2)[1:-1])))
                inner = [a for a in inner if lo < a < hi]
            # merge bands holding < 2% of training values (keeps each band estimable)
            kn = [lo] + inner + [hi]
            changed = True
            while changed and len(kn) > 2:
                changed = False
                for m in range(1, len(kn)):
                    frac = np.mean((v >= kn[m - 1]) & (v < kn[m])) if m < len(kn) - 1 else np.mean(v >= kn[m - 1])
                    if frac < 0.02:
                        kn.pop(m if m < len(kn) - 1 else m - 1); changed = True; break
            self.k[c] = kn
        self.ix_mu = {}
        for a, b in INTER:
            self.ix_mu[(a, b)] = (float(np.nanmean(X[a])), float(np.nanstd(X[a])),
                                  float(np.nanmean(X[b])), float(np.nanstd(X[b])))

    def fill(self, X):
        X = X.copy()
        for c in CONT + BIN:
            X[c] = X[c].fillna(self.med[c])
        return X

    def design(self, X, raw=None):
        """Return design matrix and column metadata. raw supplies missingness."""
        raw = X if raw is None else raw
        Xf = self.fill(X)
        cols, meta = [], []
        for c in CONT:
            kn = self.k[c]; x = Xf[c].values.astype(float)
            for m in range(1, len(kn)):
                cols.append(np.clip((x - kn[m - 1]) / (kn[m] - kn[m - 1]), 0, 1))
                meta.append((c, m, MONO[c] if self.monotone else 0))
        for c in BIN:
            cols.append(Xf[c].values.astype(float)); meta.append((c, 0, MONO[c] if self.monotone else 0))
        for c in MISS:
            cols.append(raw[c].isna().values.astype(float)); meta.append((c + '_missing', 0, 0))
        if self.interactions:
            for (a, b) in INTER:
                ma, sa, mb, sb = self.ix_mu[(a, b)]
                xa = np.clip(Xf[a].values, self.k[a][0], self.k[a][-1]) if a in self.k else Xf[a].values
                xb = np.clip(Xf[b].values, self.k[b][0], self.k[b][-1]) if b in self.k else Xf[b].values
                cols.append(((xa - ma) / sa) * ((xb - mb) / sb)); meta.append((a + '×' + b, 0, 0))
        return np.column_stack(cols), meta

    def fit(self, X, y):
        self._prep_fit(X)
        Z, self.meta = self.design(X)
        y = np.asarray(y, float); n, d = Z.shape
        lam = self.lam

        def f(theta):
            b0, w = theta[0], theta[1:]
            z = b0 + Z @ w
            p = expit(z)
            loss = np.sum(np.logaddexp(0, z) - y * z) + 0.5 * lam * w @ w
            g = Z.T @ (p - y) + lam * w
            return loss, np.concatenate([[np.sum(p - y)], g])
        bounds = [(None, None)] + [((0, None) if mono == 1 else (None, None)) for (_, _, mono) in self.meta]
        r = minimize(f, np.zeros(d + 1), jac=True, method='L-BFGS-B', bounds=bounds,
                     options={'maxiter': 5000, 'gtol': 1e-8})
        self.b0, self.w = r.x[0], r.x[1:]
        self.converged = r.success
        # baseline: mean term values on training data (for centred contributions)
        self.Zmean = Z.mean(0)
        return self

    def decision_function(self, X):
        Z, _ = self.design(X)
        return self.b0 + Z @ self.w

    def predict_proba(self, X):
        p = expit(self.decision_function(X))
        return np.column_stack([1 - p, p])

    # ---------- exact term contributions (centred) ----------
    def term_names(self):
        names = []
        for (c, _, _) in self.meta:
            if c not in names:
                names.append(c)
        return names

    def contributions(self, X):
        """Centred exact contributions: logit(x) = base + sum_t C_t(x)."""
        Z, meta = self.design(X)
        names = self.term_names()
        C = np.zeros((Z.shape[0], len(names)))
        for i, (c, _, _) in enumerate(meta):
            C[:, names.index(c)] += self.w[i] * (Z[:, i] - self.Zmean[i])
        base = self.b0 + self.Zmean @ self.w
        return base, C, names

    def shape(self, c, grid):
        """g_c(grid) for a continuous feature (main effect only), uncentred."""
        kn = self.k[c]; out = np.zeros(len(grid))
        for i, (cc, m, _) in enumerate(self.meta):
            if cc == c:
                out += self.w[i] * np.clip((grid - kn[m - 1]) / (kn[m] - kn[m - 1]), 0, 1)
        return out

    def band(self, c, v):
        kn = self.k[c]
        for m in range(1, len(kn)):
            if v < kn[m]:
                return m
        return len(kn) - 1
