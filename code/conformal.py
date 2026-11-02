"""Mondrian (group x label)-conditional split-conformal screening (CertiRisk, Algorithm 4)."""
import numpy as np


def groups(X):
    return (X['male'].values.astype(int) * 2 + (X['age'].values >= 55).astype(int))  # 0:F<55 1:F>=55 2:M<55 3:M>=55


GROUP_NAMES = ['Female, <55', 'Female, ≥55', 'Male, <55', 'Male, ≥55']


def _q(scores, eps):
    n = len(scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - eps)))
    return np.inf if k > n else np.sort(scores)[k - 1]


class Mondrian:
    def __init__(self, eps=0.10, by_group=True, by_label=True):
        self.eps, self.by_group, self.by_label = eps, by_group, by_label

    def fit(self, p_cal, y_cal, g_cal):
        self.q = {}
        G = range(4) if self.by_group else [None]
        for g in G:
            mg = (g_cal == g) if g is not None else np.ones(len(y_cal), bool)
            if self.by_label:
                for lab in (0, 1):
                    m = mg & (y_cal == lab)
                    py = p_cal[m] if lab == 1 else 1 - p_cal[m]
                    self.q[(g, lab)] = _q(1 - py, self.eps)
            else:
                py = np.where(y_cal[mg] == 1, p_cal[mg], 1 - p_cal[mg])
                self.q[(g, None)] = _q(1 - py, self.eps)
        return self

    def predict_sets(self, p, g):
        S = np.zeros((len(p), 2), bool)
        for i in range(len(p)):
            gg = g[i] if self.by_group else None
            for lab in (0, 1):
                py = p[i] if lab == 1 else 1 - p[i]
                q = self.q[(gg, lab if self.by_label else None)]
                S[i, lab] = (1 - py) <= q
        return S


def status(S):
    out = []
    for a, b in S:
        out.append('indeterminate' if (a and b) or (not a and not b) else ('positive' if b else 'negative'))
    return out
