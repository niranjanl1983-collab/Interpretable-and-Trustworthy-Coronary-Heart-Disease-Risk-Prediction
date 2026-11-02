"""Restructure the manuscript content to the requested journal flow:
Abstract, Keywords, 1 Introduction, 2 Related Work, 3 Background, 4 Proposed System,
5 Algorithm, 6 Results and Discussion, 7 Conclusion, 8 Future Scope, References.
Removes the framework name and A1-A4 labels; switches to labelled, auto-numbered tables/figures."""
import re
s = open('content.js').read()


def rep(a, b, count=1):
    global s
    assert a in s, ('MISSING', a[:90])
    s = s.replace(a, b, count)


def cut(start, end):
    """Remove and return s[start:end) given marker strings (end exclusive)."""
    global s
    i = s.index(start); j = s.index(end, i)
    seg = s[i:j]; s = s[:i] + s[j:]
    return seg

# ---------- helpers ----------
rep("img = (file, w = 1) => C.push(['img', F + file, w]),", "img = (file, w = 1, label) => C.push(['img', F + file, w, label]),")
rep("// CertiRisk manuscript (revision 2).", "// Manuscript (revision 4).")

# ---------- labels for tables ----------
TL = {'XIII': 'notation', 'XII': 'rt', 'XI': 'conf', 'X': 'inj', 'IX': 'valid', 'VIII': 'cf', 'VII': 'expl', 'VI': 'abl',
      'V': 'perf', 'IV': 'grid', 'III': 'data', 'II': 'knots', 'I': 'lit'}
s = re.sub(r"title: 'TABLE (XIII|XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)'", lambda m: f"label: '{TL[m.group(1)]}'", s)
s = re.sub(r"\bTables? (XIII|XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\b", lambda m: m.group(0).split()[0] + ' {T:' + TL[m.group(1)] + '}', s)
FL = {'1': 'arch', '2': 'ext', '3': 'calib', '4': 'shapes', '5': 'expl', '6': 'cf', '7': 'verifier', '8': 'conf', '9': 'rt'}
s = re.sub(r"Fig\. ([1-9])(?=[ab.\s),;])", lambda m: 'Fig. {F:' + FL[m.group(1)] + '}', s)
for fn, lb in [('fig_arch.png', 'arch'), ('fig_ext.png', 'ext'), ('fig_calib_dca.png', 'calib'), ('fig_shapes.png', 'shapes'), ('fig_expl.png', 'expl'),
               ('fig_cf.png', 'cf'), ('fig_verifier.png', 'verifier'), ('fig_conformal.png', 'conf'), ('fig_rt.png', 'rt')]:
    s = re.sub(r"img\('%s', ([0-9.]+)\)" % re.escape(fn), r"img('%s', \1, '%s')" % (fn, lb), s)

# ---------- title, authors, abstract, keywords ----------
i = s.index("C.push(['title'"); j = s.index("// ================= I. INTRODUCTION")
s = s[:i] + r"""C.push(['title', 'Guideline-Anchored Monotone Additive Modelling with Certified Narratives, Exact Counterfactuals and Subgroup-Conditional Conformal Screening for Ten-Year Coronary Heart Disease Risk Prediction']);
C.push(['author', 'Niranjan L¹*, Chaithra K R²']);
C.push(['affil', '¹ Department of Computer Science and Engineering (AI&ML), Nagarjuna College of Engineering and Technology, Bengaluru, Karnataka, India (affiliated to VTU, Belagavi)']);
C.push(['affil', '² Department of Computer Science and Engineering, APS College of Engineering, Bengaluru, Karnataka, India (affiliated to VTU, Belagavi)']);
C.push(['affil', '*Corresponding author: niranjanl1983@gmail.com; co-author: kr.chaithra@gmail.com']);
const mr = covRange(CM.groups, 'cov_pos');
C.push(['abstract', `Explainable cardiovascular risk tools usually explain a black-box model after the fact, so the model, its explanation and the narrative shown to the clinician can disagree. This paper proposes an interpretable-by-design framework in which every explanation is exact and every sentence of the risk narrative can be re-derived from the model. The framework combines a guideline-anchored monotone additive model (GAMA), whose risk curves have knots at clinical cut-points for blood pressure, cholesterol and glucose and cannot run against established risk directions; a certified narrative compiler, which attaches a checkable certificate (value, band, contribution, odds multiplier, rank and source) to every statement, with an independent verifier; an exact minimum-cost counterfactual search; and subgroup-conditional (Mondrian) conformal screening. On the public Framingham dataset (${N} participants, ${A.events} ten-year coronary events) under repeated nested cross-validation, GAMA reached a ROC-AUC of ${auc('GAMA')} and ranked third of fourteen systems. It was statistically tied with logistic regression, an explainable boosting machine and extra trees, and significantly better than the other ten classifiers, including XGBoost and LightGBM. It showed no monotonicity violations (LightGBM: ${pc(VI.LightGBM.rate / 100)}%) and the most stable attributions (Spearman ρ ${f(ST.GAMA.rho)}). The counterfactual search was never undercut by random or greedy search. The verifier blocked ${VT.full.toLocaleString('en-US')} of ${VT.n.toLocaleString('en-US')} injected narrative errors, against ${pc(VT.bind / VT.n)}% for value–concept binding, without false alarms. Conformal screening held coverage among coronary events at ${pc(mr[0])}–${pc(mr[1])}% in every sex and age subgroup, versus ${pc(CMg.cov_pos)}% for marginal conformal prediction. The complete pipeline ran in a median of ${f(RT.negative.total.median, 0)}–${f(RT.positive.total.median, 0)} ms per patient. Exact, verifiable explanation therefore cost no discrimination; clinical value requires external and prospective validation.`]);
C.push(['keywords', 'Explainable artificial intelligence; Cardiovascular risk prediction; Monotone additive model; Counterfactual explanation; Conformal prediction']);

""" + s[j:]

# ---------- 1. Introduction ----------
rep("h1('I. INTRODUCTION');", "h1('1. Introduction');")
rep("To answer them we propose CertiRisk, built on the principle of", "To answer them we propose a framework built on the principle of")
rep("Its first contribution is GAMA,", "Its first contribution is GAMA (Algorithm 1),")
rep("The second contribution is a certified narrative compiler and an independent certificate verifier.", "The second contribution is a certified narrative compiler and an independent certificate verifier (Algorithm 2).")
rep("The third contribution is EMC, an exact minimum-cost counterfactual search.", "The third contribution is an exact minimum-cost counterfactual search, referred to as EMC (Algorithm 3).")
rep("The fourth contribution is a Mondrian conformal screening layer", "The fourth contribution is a Mondrian conformal screening layer (Algorithm 4)")
i = s.index("p('The remainder of the paper is organised as follows."); j = s.index("\n", i)
s = s[:i] + "p('The remainder of the paper is organised as follows. Section 2 reviews related work and summarises it in a comparative table. Section 3 introduces the clinical and methodological background on which the framework builds. Section 4 presents the proposed system, and Section 5 gives its algorithms with their complexity. Section 6 describes the dataset and experimental design and reports and discusses the results. Section 7 concludes, and Section 8 outlines the future scope of the work. The principal notation is listed in the nomenclature table at the end of the paper.');" + s[j:]

# ---------- 2. Related Work ----------
rep("h1('II. RELATED WORK');", "h1('2. Related Work');")
rep("as explained in Section I,", "as explained in Section 1,")
rep("['**CertiRisk (this work)**',", "['**Proposed framework (this work)**',")

# ---------- 3. Background (new) ----------
BACK = r"""// ================= 3. BACKGROUND =================
h1('3. Background');
p('This section summarises the clinical thresholds and the four methodological foundations—additive logistic modelling, post-hoc feature attribution, counterfactual explanation and conformal prediction—on which the proposed system builds. The notation introduced here is used throughout Sections 4 and 5.');
h2('3.1 Clinical Risk Factors and Guideline Thresholds');
p('Clinical guidelines translate continuous measurements into categories that drive decisions. For blood pressure, the 2017 ACC/AHA guideline defines normal (<120/<80 mmHg), elevated (120–129/<80 mmHg), stage 1 (130–139 or 80–89 mmHg) and stage 2 (≥140 or ≥90 mmHg) hypertension {whelton2018}, whereas the earlier JNC 7 report places stage 1 at 140–159/90–99 mmHg and stage 2 at ≥160/≥100 mmHg {chobanian2003}. The ATP III report classifies total cholesterol as desirable (<200 mg/dL), borderline high (200–239 mg/dL) or high (≥240 mg/dL) {atp3}, and the ADA standards define fasting plasma glucose of 100–125 mg/dL as the prediabetes range and ≥126 mg/dL as the diabetes range {ada2024}. Body-mass index is classified by the WHO as underweight (<18.5), normal (18.5–24.9), overweight (25–29.9) or obese (≥30 kg/m²) {who2000}. Table {T:guide} lists these categories. Two features of the table matter for this work: the same measurement can fall into different categories under different editions, so every category must be reported with its source, and a glucose category requires a verified fasting sample, which the Framingham teaching dataset does not record.');
tbl({ label: 'guide', caption: 'Clinical Guideline Categories Used by the Proposed System', widths: [2000, 2300, 5338], headers: ['Measurement', 'Source', 'Categories'], rows: [
  ['Blood pressure (mmHg)', 'ACC/AHA 2017 {whelton2018}', 'Normal < 120/< 80; elevated 120–129/< 80; stage 1 130–139 or 80–89; stage 2 ≥ 140 or ≥ 90'],
  ['Blood pressure (mmHg)', 'JNC 7 (2003) {chobanian2003}', 'Normal < 120/< 80; prehypertension 120–139 or 80–89; stage 1 140–159 or 90–99; stage 2 ≥ 160 or ≥ 100'],
  ['Total cholesterol (mg/dL)', 'ATP III (2002) {atp3}', 'Desirable < 200; borderline high 200–239; high ≥ 240'],
  ['Fasting glucose (mg/dL)', 'ADA 2024 {ada2024}', 'Normal < 100; prediabetes range 100–125; diabetes range ≥ 126 (requires confirmed fasting)'],
  ['Body-mass index (kg/m²)', 'WHO (2000) {who2000}', 'Underweight < 18.5; normal 18.5–24.9; overweight 25–29.9; obesity ≥ 30'],
], note: 'The 2017 ACC/AHA guideline has since been superseded {jones2025}; it is used because its cut-points define the narrative bands, and every category is reported with its edition.' });
h2('3.2 Additive Logistic Models');
p('Let x ∈ ℝᵈ denote a patient record and y ∈ {0, 1} the ten-year CHD outcome. Logistic regression models the log-odds as a linear function of the inputs, and a generalized additive model (GAM) replaces each linear term with a smooth univariate shape function gⱼ {hastie1986}:');
eq('\\operatorname{logit}\\Pr(Y=1\\mid\\mathbf{x})=\\log\\frac{\\Pr(Y=1\\mid\\mathbf{x})}{1-\\Pr(Y=1\\mid\\mathbf{x})}=b_0+\\sum_{j=1}^{d} g_j(x_j)');
p('Because the log-odds is a sum of per-feature terms, the contribution of each feature to an individual prediction is exactly gⱼ(xⱼ), up to a constant. Additive models therefore provide explanations that are exact rather than approximate {lou2012,caruana2015}. A shape function is monotone non-decreasing if gⱼ(a) ≤ gⱼ(b) whenever a ≤ b. Imposing this property on risk factors with an established adverse direction prevents the model from learning, for example, that higher blood pressure lowers risk {gupta2016}.');
h2('3.3 Post-Hoc Feature Attribution');
p('For a black-box model f, the Shapley value of feature i averages its marginal contribution over all subsets S of the other features {lundberg2017}:');
eq('\\phi_i(f,\\mathbf{x})=\\sum_{S\\subseteq F\\setminus\\{i\\}}\\frac{|S|!\\,(|F|-|S|-1)!}{|F|!}\\Big[f_{\\mathbf{x}}(S\\cup\\{i\\})-f_{\\mathbf{x}}(S)\\Big]');
p('where fₓ(S) is the expected model output when only the features in S are fixed at their observed values. The value of fₓ(S) depends on how the absent features are filled in, so different estimators give different attributions when features are correlated {aas2021}. LIME instead fits a weighted linear surrogate to perturbed samples around x {ribeiro2016}. Both are approximations of the model, which is the gap that an additive model closes by construction.');
h2('3.4 Counterfactual Explanations');
p('A counterfactual explanation for a record x classified as positive is a nearby record x′ that the model classifies as negative {wachter2018}:');
eq('\\mathbf{x}^{\\mathrm{cf}}=\\arg\\min_{\\mathbf{x}\'\\in\\mathcal{A}(\\mathbf{x})}\\ d(\\mathbf{x},\\mathbf{x}\')\\quad\\text{s.t.}\\quad f(\\mathbf{x}\')<t^{*}');
p('where d is a cost function, 𝒜(x) the set of admissible changes and t* the decision threshold. For general models the problem is non-convex and is solved heuristically {mothilal2020}; for linear models it can be solved exactly by integer programming {ustun2019}. A counterfactual describes the model and not the causal effect of an intervention {karimi2021}.');
h2('3.5 Conformal Prediction');
p('Split conformal prediction turns any probabilistic classifier into a set-valued predictor with a finite-sample coverage guarantee {vovk2005,angelopoulos2023}. Given n calibration records with non-conformity scores αᵢ = 1 − p̂(yᵢ | xᵢ), let q̂ be the ⌈(n + 1)(1 − ε)⌉-th smallest score. Then the set Γ(x) = {y : 1 − p̂(y | x) ≤ q̂} satisfies');
eq('\\Pr\\big(Y\\in\\Gamma(\\mathbf{X})\\big)\\ge 1-\\varepsilon');
p('provided calibration and test records are exchangeable. This guarantee is marginal: it holds on average over all patients and can fail badly for a minority class or subgroup. Mondrian conformal prediction computes a separate threshold within each category of a prespecified partition and guarantees coverage within each category {vovk2005,sadinle2019}.');

"""
rep("// ================= III. PROPOSED FRAMEWORK =================", BACK + "// ================= 4. PROPOSED SYSTEM =================")

# ---------- 4. Proposed System ----------
rep("h1('III. PROPOSED CERTIRISK FRAMEWORK');", "h1('4. Proposed System');\nh2('4.1 System Architecture');")
rep("p('Fig. {F:arch} shows the CertiRisk pipeline.", "p('Fig. {F:arch} shows the architecture of the proposed system.")
rep("The record enters GAMA (A1), which produces an exact decomposition of the log-odds. The exact counterfactual search (A3) and the conformal layer (A4) operate only on the frozen GAMA model. The narrative compiler (A2) assembles",
    "The record enters GAMA, which produces an exact decomposition of the log-odds. The exact counterfactual search and the conformal layer operate only on the frozen GAMA model. The narrative compiler assembles")
rep("cap('**Fig. {F:arch}.** CertiRisk pipeline. A1–A4 denote the four algorithms.", "cap('**Fig. {F:arch}.** Architecture of the proposed system; the numbers refer to Algorithms 1–4.")
rep("h2('A. Guideline-Anchored Monotone Additive Model (A1)');", "h2('4.2 Guideline-Anchored Monotone Additive Model');")
rep("did not improve discrimination (Section V-A)", "did not improve discrimination (Section 6.4)")
rep("Algorithm 1 summarises the fitting procedure.');", "The fitting procedure is given in Algorithm 1 (Section 5.1).');")
rep("h2('B. Certified Narrative Compilation and Verification (A2)');", "h2('4.3 Certified Narrative Compilation and Verification');")
rep("Algorithm 2 summarises the verifier, and Box 1 in Section V shows a compiled narrative.');", "Algorithm 2 (Section 5.2) gives the compilation and verification procedures, and Box 1 in Section 6.8 shows a compiled narrative.');")
rep("h2('C. Exact Counterfactual Search and Conformal Screening (A3, A4)');", "h2('4.4 Exact Counterfactual Search');")
rep("p('The conformal layer uses a selection set", "h2('4.5 Subgroup-Conditional Conformal Screening');\np('The conformal layer uses a selection set")
rep("The narrative reports the status and the verifier checks it. Algorithm 3 summarises EMC.');", "The narrative reports the status and the verifier checks it. Algorithms 3 and 4 (Sections 5.3 and 5.4) give the counterfactual search and the conformal procedure.');")
rep("(Proposition 3).');", "(Proposition 3).');")

# move algorithm blocks out of section 4
a1 = cut("alg('Algorithm 1: GAMA fitting (A1)'", "\nh2('4.3 Certified")
a2 = cut("alg('Algorithm 2: Certificate verification (A2)'", "\nh2('4.4 Exact")
a3 = cut("alg('Algorithm 3: Exact monotone counterfactual search (A3)'", "\n// ================= IV. EXPERIMENTAL SETUP")
a1 = a1.replace("Algorithm 1: GAMA fitting (A1)", "Algorithm 1: Guideline-anchored monotone additive model (GAMA) fitting")
a2 = a2.replace("Algorithm 2: Certificate verification (A2)", "Algorithm 2 (b): Certificate verification")
a3 = a3.replace("Algorithm 3: Exact monotone counterfactual search (A3)", "Algorithm 3: Exact minimum-cost counterfactual search (EMC)")

ALG = r"""// ================= 5. ALGORITHM =================
h1('5. Algorithm');
p('This section gives the four algorithms of the proposed system in pseudocode, followed by the end-to-end procedure that is run for each patient. For each algorithm we state its inputs, outputs and computational cost. Throughout, N is the number of training records, P the number of model parameters (here P = 37), m the number of grid values per modifiable feature and K = 6 the number of modifiable features.');
h2('5.1 Algorithm 1: GAMA Fitting');
p('Algorithm 1 places the knots, builds the ramp basis and solves the box-constrained logistic problem of Eq. (6). Steps 1–3 take O(Nd log N) time for the percentile computations, and each L-BFGS-B iteration costs O(NP). In our experiments a complete fit, including knot placement, took about 20 ms. The output is a frozen model whose contributions cₜ(x) are exact by Eq. (7).');
""" + a1 + r"""
h2('5.2 Algorithm 2: Certified Narrative Compilation and Verification');
p('Algorithm 2 has two procedures. The compiler (part a) turns a prediction into a narrative in a closed grammar and attaches a certificate to every statement. The verifier (part b) receives only the narrative text, the record and the frozen model, and re-derives every certified quantity using the tolerances of Eq. (9). Both procedures are linear in the number of sentences and require one model evaluation (plus one for a stated counterfactual), so their cost is dominated by parsing. Compilation and verification each took about 3–5 ms per patient (Section 6.10).');
alg('Algorithm 2 (a): Certified narrative compilation', [
  '**Input:** record x, frozen model f, threshold t*, registry R, counterfactual x′ (optional), conformal status s   **Output:** narrative N',
  '1: (β, c, z) ← f.contributions(x);  p ← σ(z);  top ← k main-effect terms with largest |c|',
  '2: emit probability p and threshold t*; emit screen = positive if p ≥ t*, else negative; emit conformal status s',
  '3: emit global certificate: β, L = Σ_{t∈top} cₜ, O = z − β − L, z',
  '4: for r = 1..k: t ← top[r]; emit value, unit, band Bₜ(xₜ), direction(cₜ), |cₜ|, exp(cₜ), rank r, term ID',
  '5: emit strongest contributor = top[1]',
  '6: for each registry rule (ACC/AHA, JNC 7, ATP III, ADA, WHO): emit category with source ID; ADA → "no category" if fasting unknown',
  '7: if p ≥ t*: emit x′ (changed features, values, re-scored probability) and the model-not-treatment disclaimer, or "no admissible change"',
  '8: emit scope sentence; **return** N',
]);
""" + a2 + r"""
h2('5.3 Algorithm 3: Exact Minimum-Cost Counterfactual Search');
p('Algorithm 3 solves Eq. (10) on the admissible grid. Step 2 evaluates the model once for each of the at most Km grid values. Single changes are found in O(Km) time, and each of the K(K − 1)/2 feature pairs in O(m log m) time by sorting and binary search, giving O(K²m log m) in total. The coupled pair of systolic and diastolic blood pressure is enumerated jointly in O(m²). Every candidate returned is re-scored by the full model, so the guarantee of Proposition 3 does not depend on the precomputed reductions. The median run time was about 30 ms per screen-positive patient.');
""" + a3 + r"""
h2('5.4 Algorithm 4: Mondrian Conformal Screening');
p('Algorithm 4 calibrates one threshold for each of the eight cells defined by sex, age band (below or at least 55 years) and outcome label, using a selection set that is disjoint from GAMA’s training data, and then forms the prediction set of Eq. (12) for a new patient. Calibration costs O(n log n) for n selection records, and screening a patient costs O(1).');
alg('Algorithm 4: Mondrian (group × label) conformal calibration and screening', [
  '**Input:** frozen model f, selection set {(xᵢ, yᵢ)}, groups g(·), miscoverage ε   **Output:** status for a new record x',
  '1: for each selection record i: αᵢ ← 1 − p̂(yᵢ | xᵢ)',
  '2: for each group g and label y: S ← {αᵢ : g(xᵢ) = g, yᵢ = y};  n ← |S|',
  '3:    q̂(g, y) ← the ⌈(n + 1)(1 − ε)⌉-th smallest value of S  (∞ if that rank exceeds n)',
  '4: for a new record x: Γ ← {y ∈ {0, 1} : 1 − p̂(y | x) ≤ q̂(g(x), y)}',
  '5: **return** positive if Γ = {1};  negative if Γ = {0};  indeterminate otherwise',
]);
h2('5.5 End-to-End Procedure per Patient');
p('Algorithm 5 combines the four algorithms into the procedure executed for every patient at inference time. Only the frozen model, the calibrated conformal thresholds and the guideline registry are needed; the training data are not accessed.');
alg('Algorithm 5: Per-patient inference with certified narrative', [
  '**Input:** record x; frozen GAMA f (Algorithm 1); conformal thresholds q̂ (Algorithm 4); threshold t*; registry R   **Output:** verified narrative or expert-review flag',
  '1: p ← f(x);  (β, c, z) ← exact contributions of x',
  '2: s ← conformal status of x (Algorithm 4, step 4–5)',
  '3: if p ≥ t*: x′ ← EMC(x, f, t* − δ) (Algorithm 3)',
  '4: N ← Compile(x, f, t*, R, x′, s) (Algorithm 2a)',
  '5: reasons ← Verify(N, x, f, t*, s) (Algorithm 2b)',
  '6: **return** N as a software-checked draft if reasons = ∅, else route N to expert review with reasons',
]);

"""
rep("// ================= IV. EXPERIMENTAL SETUP =================", ALG + "// ================= 6. RESULTS AND DISCUSSION =================")

# ---------- 6. Results and Discussion ----------
rep("h1('IV. EXPERIMENTAL SETUP');", "h1('6. Results and Discussion');\np('This section first describes the dataset and the experimental design (Sections 6.1–6.3) and then reports the results (Sections 6.4–6.10), followed by a discussion and the limitations of the study (Sections 6.11–6.12).');")
rep("h2('A. Dataset Description and Preprocessing');", "h2('6.1 Dataset Description');")
rep("h2('B. Validation Strategy and Performance Metrics');", "h2('6.2 Validation Strategy and Evaluation Metrics');")
rep("h2('C. Interpretability Assessment Design');", "h2('6.3 Design of the Interpretability, Recourse, Verification and Real-Time Studies');")
rep("h2('D. Design of the Recourse, Verification and Uncertainty Studies');\n", "")
rep("h2('E. Real-Time Execution Benchmark');\n", "")
rep("The third verifier is the full CertiRisk certificate verifier.", "The third verifier is the full certificate verifier of the proposed system.")
rep("p(`To assess whether CertiRisk can run at the point of care,", "p(`To assess whether the proposed system can run at the point of care,")
rep("h1('V. RESULTS');\n", "")
rep("h2('A. Comparative Predictive Performance');", "h2('6.4 Comparative Predictive Performance');")
rep("h2('B. Learned Risk Profiles and Clinical Consistency');", "h2('6.5 Learned Risk Profiles and Clinical Consistency');")
rep("as discussed in Section VI.", "as discussed in Section 6.11.")
rep("h2('C. Robustness of Feature Attributions');", "h2('6.6 Robustness of Feature Attributions');")
rep("h2('D. Optimal Actionable Recourse');", "h2('6.7 Optimal Actionable Recourse');")
rep("h2('E. Narrative Fidelity Under Error Injection');", "h2('6.8 Narrative Fidelity Under Error Injection');")
rep("'CertiRisk accepted'", "'Proposed verifier accepted'")
rep("'CertiRisk certificate'", "'Proposed certificate verifier'")
rep("injections accepted by CertiRisk were verified", "injections accepted by the proposed verifier were verified")
rep("h2('F. Subgroup-Reliable Uncertainty Quantification');", "h2('6.9 Subgroup-Reliable Uncertainty Quantification');")
rep("h2('G. Real-Time Feasibility');", "h2('6.10 Real-Time Feasibility');")
rep("caption: 'Per-Patient Latency of the CertiRisk Pipeline (ms)'", "caption: 'Per-Patient Latency of the Proposed Pipeline (ms)'")
rep("h1('VI. DISCUSSION');", "h2('6.11 Discussion');")
rep("p('The study has several limitations.", "h2('6.12 Limitations');\np('The study has several limitations.")
rep("// ================= VI. DISCUSSION =================\n", "")

# ---------- 7. Conclusion, 8. Future Scope ----------
i = s.index("h1('VII. CONCLUSION AND FUTURE SCOPE');"); j = s.index("h2('Ethics Statement');")
decl_marker = s[j:]
k = decl_marker.index("// ================= NOTATION (end) =================")
decl = decl_marker[:k]
notation = decl_marker[k:]
s = s[:i] + r"""h1('7. Conclusion');
p(`This paper presented an interpretable-by-design framework for ten-year coronary heart disease risk that replaces post-hoc explanation and free-text verification with four connected algorithms: a guideline-anchored monotone additive model, a certified narrative compiler and verifier, an exact minimum-cost counterfactual search and Mondrian conformal screening. The design principle is that every statement shown to a clinician—each contribution, its magnitude and rank, each guideline category, each suggested change and each uncertainty status—is derived from a single interpretable model and can be re-checked against it by a simple program.`);
p(`On the public Framingham dataset, under leakage-free repeated nested cross-validation, GAMA reached a ROC-AUC of ${f(T.GAMA.AUC)}, ranked third of fourteen systems, was statistically tied with the best, and significantly outperformed ten classifiers, including XGBoost and LightGBM. It produced no monotonicity violations and the most stable attributions. The counterfactual search was provably optimal and was never undercut by baseline searches. The verifier blocked every injected statement that was actually false, including the ${pc(1 - VT.bind / VT.n, 0)}% that a value-binding gate missed, with no false alarms on meaning-preserving rewrites. Conformal screening kept coverage near the nominal 90% in every sex, age and outcome subgroup, and the complete pipeline ran in tens of milliseconds per patient. These results show that, for tabular cardiovascular risk prediction, exact and verifiable explanation need not cost discrimination.`);

// ================= 8. FUTURE SCOPE =================
h1('8. Future Scope');
p('Several directions follow from this work. First, the framework should be validated externally and prospectively on contemporary cohorts that record the inputs the Framingham teaching dataset lacks—HDL cholesterol, kidney function, verified fasting glucose and ethnicity—so that guideline risk equations such as the Pooled Cohort Equations can be computed and compared with the model. Validation on Indian and other South Asian populations is a priority, because Western risk equations are known to transport poorly to them.');
p('Second, the certified narrative can be made more fluent without losing verifiability. A language model could paraphrase the compiled narrative under a grammar that maps each generated sentence back to its certificate, so that the verifier continues to check every claim. The same mechanism would support patient-facing and multilingual narratives, for example in Kannada and Hindi, with verification performed on a language-independent certificate.');
p('Third, the counterfactual search can be extended to larger change budgets by dynamic programming over discretised log-odds reductions, to clinician-specified cost functions, and to causal constraints that link, for instance, weight loss to blood-pressure change. Fourth, knot positions and monotonicity priors could be learned or adapted per population under clinician review, and the conformal layer extended to finer subgroups as larger cohorts become available.');
p('Finally, the clinical value of certified narratives must be tested with their intended users. A randomised reader study with physicians should measure comprehension, detection of erroneous model output and over-reliance, followed by a silent prospective deployment that monitors calibration drift, verifier block rates and latency before any clinical use.');

// ================= NOMENCLATURE =================
""" + notation.replace("// ================= NOTATION (end) =================\n", "").replace("h1('APPENDIX: PRINCIPAL NOTATION');", "h1('Nomenclature');") \
  .replace("caption: 'Principal Notation'", "caption: 'Principal Notation'").replace("\nmodule.exports = C;", "") + r"""
// ================= DECLARATIONS =================
h1('Declarations');
""" + decl + "\nmodule.exports = C;\n"

# remaining CertiRisk mentions
assert 'CertiRisk' not in s.replace('// Manuscript', ''), [m.start() for m in re.finditer('CertiRisk', s)]
for a in ['(A1)', '(A2)', '(A3)', '(A4)', 'A1–A4']:
    assert a not in s, a
open('content.js', 'w').write(s)
print('ok')
