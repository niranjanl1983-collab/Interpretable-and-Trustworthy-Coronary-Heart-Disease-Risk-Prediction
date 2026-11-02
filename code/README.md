# CertiRisk — code and results

Data: public Framingham teaching dataset (`framingham.csv`, 4,240 rows; SHA-256 2c0e57dc…fadb018), placed at the path in `common.py` (DATA).

Algorithms
- `gama.py` — A1 GAMA (guideline-anchored monotone additive model)
- `narrative.py` — A2 certified narrative compiler, certificate verifier, and baselines B1 (numeric membership) / B2 (value–concept binding)
- `emc.py` — A3 exact minimum-cost counterfactual search, plus random-search and greedy baselines
- `conformal.py` — A4 Mondrian (sex × age × label) conformal screening

Experiments (run in this order)
1. `exp1_cv.py` — repeated nested CV, all models (then `exp1_partial.py GAMA "GAM (unconstrained)" "GAMA (quantile knots)"` re-runs the interaction-free final GAMA and ablations on identical folds)
2. `exp2_expl.py` — monotonicity violations and attribution stability
3. `exp_cert.py` — counterfactual benchmark, narrative verification challenge, conformal coverage
4. `analyze1.py`, `analyze2.py`, `fig_arch.py` — tables (JSON) and figures

Python 3.13; numpy 2.5, pandas 3.0, scipy 1.18, scikit-learn 1.9, lightgbm 4.7, interpret 0.7, shap 0.54, matplotlib.
