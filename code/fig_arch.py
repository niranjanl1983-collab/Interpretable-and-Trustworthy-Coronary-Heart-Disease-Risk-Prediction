import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from analyze1_style import *

fig, ax = plt.subplots(figsize=(7.4, 4.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 70); ax.axis('off')
INK, MUT, BL, OR, AQ, VI, GR = '#0b0b0b', '#52514e', '#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#8a8984'


def box(x, y, w, h, title, body, c, fs=6.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.2', fc=c + '18', ec=c, lw=1.3))
    ax.text(x + w / 2, y + h - 1.6, title, ha='center', va='top', fontsize=7.6, weight='bold', color=INK)
    ax.text(x + w / 2, y + h - 6.0, body, ha='center', va='top', fontsize=fs, color=MUT, linespacing=1.4)


def arrow(pts, txt=None, tpos=None):
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:-1]):
        ax.plot([x1, x2], [y1, y2], color=MUT, lw=1.0)
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', color=MUT, lw=1.0, mutation_scale=8))
    if txt: ax.text(*tpos, txt, ha='center', fontsize=6, color=MUT, style='italic')

box(1, 47, 14, 21, 'Record x', '15 predictors\nmissingness\nindicators', GR)
box(19, 47, 24, 21, 'GAMA (Algorithm 1)', 'knots at guideline cut-points:\nSBP 120/130/140/160\ncholesterol 200/240\nglucose 100/126\nmonotone bands (w ≥ 0)', BL)
box(47, 47, 20, 21, 'EMC (Algorithm 3)', 'exact minimum-cost\ncounterfactual,\n≤ 2 modifiable\nmeasurements,\nre-scored by model', AQ)
box(19, 22, 24, 20, 'Guideline registry', 'ACC/AHA 2017 BP, JNC 7\nATP III cholesterol\nADA 2024 (fasting unverified:\nno category), WHO BMI', GR)
box(47, 29, 20, 13, 'Conformal (Alg. 4)', 'Mondrian sets by\nsex × age × label', VI)
box(71, 22, 28, 46, 'Certified narrative\ncompiler (Algorithm 2a)', '\nevery sentence carries\na certificate:\n\nterm · value · unit\nguideline band\nlog-odds contribution\nodds multiplier · rank\nrule / source ID\n\nno free prose; no\noutcome label in input', OR)
box(1, 1, 98, 16, 'Certificate verifier (Algorithm 2b)  —  input: narrative text, record, frozen model only',
    'Re-derives every certified quantity: baseline + Σ contributions = log-odds → probability; values, units, bands;\n'
    'directions, magnitudes, odds multipliers, ranks; guideline categories, source IDs; counterfactual re-scored;\n'
    'conformal status; scope sentence; forbidden claims.  All pass → software-checked draft; any failure → expert review.', INK, fs=6.1)
arrow([(15.4, 57.5), (18.6, 57.5)])
arrow([(43.4, 60), (46.6, 60)], 'z, cⱼ', (45, 61.3))
arrow([(43.4, 52), (45, 52), (45, 36), (46.6, 36)], 'p(x)', (42.6, 44))
arrow([(67.4, 57.5), (70.6, 57.5)])
arrow([(67.4, 35.5), (70.6, 35.5)])
arrow([(43.4, 25.5), (70.6, 25.5)], 'categories + source IDs', (57, 26.6))
arrow([(85, 21.6), (85, 17.4)])
plt.savefig('/home/claude/p3/figs/fig_arch.png', bbox_inches='tight'); plt.close()
