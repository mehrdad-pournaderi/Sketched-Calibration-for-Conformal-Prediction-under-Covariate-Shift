# Sketched Calibration for Conformal Prediction under Covariate Shift

Code, results, logs and manuscript source for

> M. Pournaderi. *Sketched Calibration for Conformal Prediction under Covariate Shift.* 2026.

Weighted conformal prediction corrects for covariate shift by reweighting calibration scores
with the likelihood ratio dQ_X/dP_X, at a calibration cost governed by the chi-square divergence
between the two covariate laws. *Sketched calibration* runs the weighted procedure with the
ratio of a compressed covariate Z = T(X), either on a score fitted to Z or, keeping a predictor on
all covariates, in the weights alone. The shift-dependent calibration cost cannot increase, and
the coverage guarantee becomes 1 − α − Δ_T, where the *leakage* Δ_T is the part of the discarded
shift that reappears as a change in the law of the response (or of the score) given Z. The paper
characterises the leakage (it is a covariance between the discarded shift and the response on the
fibres of the sketch), proves that no threshold rule can avoid it, makes everything explicit in a
Gaussian single-index model, discusses what can be certified about a sketch, and reports the
experiments reproduced here.

## Layout

```
code/
  sketched_calibration_experiments.py   E1–E5 (Tables 1–5), Gaussian single-index model
  fig_hetero.py, make_fig.py            E1 on a finer shift grid -> results/fig_hetero.json -> paper/fig_hetero.pdf (Figure 1)
  sc_lib.py                             shared helpers (weighted threshold, SIR, reporting)
  exp_E6.py                             E6: full-covariate score with sketched weights (Table 6)
  exp_E7.py                             E7: nonlinear multiple-index model, learned sketches, selection (Table 7, Figure 2)
  exp_E8.py                             E8: California housing and UCI protein data with exponential tilting (Tables 8–9)
  make_fig2.py                          Figure 2 from results/exp_E7_*.json
  make_tables.py                        LaTeX rows of Tables 7–9 from the JSON results
results/                                JSON results of every run reported in the paper
logs/                                   raw console output of those runs
data/download_data.sh                   downloads CASP.csv (UCI protein data); California housing is fetched by scikit-learn
paper/                                  manuscript source (pdflatex + bibtex), figures, compiled PDF
```

## Experiments

| Name | Setting | Paper |
|---|---|---|
| E1 | informed sketches Π₁ (k = 1), Π₈ (k = 8); heteroscedastic noise; shift with component 0.7 along the index | Figure 1, Table 1 |
| E2 | E1 with ratios estimated by logistic regression on unlabelled covariates | Table 2 |
| E3 | random sketch k = 8, random shift direction, homoscedastic noise | Table 3 |
| E4 | shift along the signal, sweep over the sketch dimension k | Table 4 |
| E5 | informed sketches, homoscedastic noise, random shift direction (control) | Table 5 |
| E6 | full-covariate predictor with sketched weights, settings of E1 and E3 | Table 6 |
| E7 | nonlinear multiple-index model, gradient-boosting predictor, learned sketches, selection with 100 target labels | Figure 2, Table 7 |
| E8 | California housing and protein (CASP) data with known exponential tilts | Tables 8–9 |

## Reproducing

```
pip install -r requirements.txt
python code/sketched_calibration_experiments.py E1      # or E2 … E5, or all
python code/fig_hetero.py && python code/make_fig.py    # Figure 1
python code/exp_E6.py all                               # Table 6 (about a minute)
for i in 0 1 2 3 4 5 6; do python code/exp_E7.py $i; done   # Table 7 / Figure 2 (about 1.5 h single-threaded)
python code/make_fig2.py
bash data/download_data.sh
python code/exp_E8.py california all; python code/exp_E8.py protein all   # Tables 8–9 (about 2 h)
python code/make_tables.py                              # LaTeX rows for Tables 7–9
cd paper && pdflatex sketched_calibration && bibtex sketched_calibration \
         && pdflatex sketched_calibration && pdflatex sketched_calibration
```

All experiments are seeded with `numpy.random.default_rng`, so re-running reproduces the logged
numbers up to floating-point differences across BLAS builds (E7–E8 use scikit-learn gradient
boosting with `random_state=0`). The reported numbers were produced with numpy 2.5.3, scipy
1.18.1, scikit-learn 1.9.1, pandas (any 2.x), matplotlib 3.11.2.

## License

MIT (see `LICENSE`).
