# Python Port Notes

`runfit.py` is a Python port of the MATLAB fit/check workflow in `runfit.m`.
It is aligned with the notation in `Analysis.pdf` and `Numerics.pdf`:

- `omega` is loaded from `data.mat` variable `w`.
- `zeta = theta/x1` is loaded from `data.mat` variable `v`.
- `u1,u2` are velocity components loaded from `Vel`.
- `Fomega` and `Fzeta` are the steady dynamic-rescaling residuals after writing
  `theta = x1*zeta`.

## Dependencies

```powershell
python -m pip install -r requirements.txt
```

or, without touching the active Python environment:

```powershell
$cp = python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"
python -m pip install --target ".python_deps\$cp" -r requirements.txt
```

The scripts check for a compatible versioned folder such as `.python_deps\cp314`,
`.python_deps_fresh\cp314`, or `.python_deps\cp312` and prepend it to
`sys.path`. This matters because NumPy/SciPy wheels compiled for one Python
version cannot be imported by another. The `_fresh` folder is only a fallback
for interrupted or permission-damaged target installs.

The corresponding native MATLAB optimizer is `optimize_rank_factors.m`:

```powershell
matlab -batch "cd('C:\Users\Fan\Documents\Datafit'); optimize_rank_factors(20);"
```

## Reading MATLAB Data From Python

`data.mat` is a MATLAB v5 MAT file. It is not a MATLAB `-v7.3` HDF5 file. The
Python port therefore uses SciPy:

```python
from scipy.io import loadmat

names = ["x1", "x2", "w", "v", "Vel", "BS1d_large"]
data = loadmat("data.mat", variable_names=names, simplify_cells=True)

x1 = data["x1"]                 # NumPy array
omega = data["w"]               # NumPy array
zeta = data["v"]                # NumPy array, zeta = theta/x1
u1 = data["Vel"]["u1"]          # MATLAB struct -> Python dict
bs0 = data["BS1d_large"][0]     # MATLAB sparse -> SciPy sparse array
```

The important options are:

- `variable_names=...` loads only the variables the Python code needs.
- `simplify_cells=True` converts the saved MATLAB struct `Vel` into a Python
  dictionary and squeezes simple vectors into one-dimensional arrays.

MATLAB function handles are the exception. `data.mat` stores function handles
named `AG` and `Chi20`; SciPy can deserialize their containers but cannot
execute MATLAB function handles. The Python port rebuilds them from the source
formulas:

- `chi20_derivative` implements the exponential cutoff derivatives from
  `Build_fun1D.m` and `Cutoff_exp_modi.m`.
- `ag_jets` implements the angular-coefficient recurrence from
  `Build_deri_r_ag.m`.

For a `-v7.3` MAT file, use an HDF5 reader such as `h5py` or a v7.3-aware MAT
loader instead of `scipy.io.loadmat`.

## Implemented Workflow

`runfit.py` implements the same numerical path as `runfit.m`:

1. Load numeric arrays, sparse B-spline matrices, and the velocity struct from
   `data.mat`.
2. Fit `omega` and `zeta` with an SVD plus sixth-order B-spline remesh.
3. Fit the stored velocity pieces `u10f` and `u20f` using damped versions of the
   same SVD/B-spline routine.
4. Rebuild the semi-analytic far-field stream-function contribution `Psi1`.
5. Assemble `u1,u2` and their first derivatives.
6. Print field-fit checks, velocity consistency checks, and `Fomega`/`Fzeta`
   residual checks.

The B-spline code mirrors Appendix C of `Numerics.pdf`: reflected odd/even
basis functions at `x1=0`, seventh-order extrapolation near the boundary and
far field, and derivative matrices built from the same basis functions.

## Optimization Scripts

The current optimization stack uses one shared explicit-`S` rank-factor model:

- `rank_optimization_model.py`: objective, explicit `P,Q,s,c_l,c_omega`
  variables, analytic gradients, fixed-gauge tangent projection, and
  retraction/refit integration.
- `gradient_check_rank.py`: finite-difference check for the shared analytic
  gradients.
- `check_retraction_refit.py`: verifies that the QR/SVD retraction and B-spline
  refit preserve the fitted cores.
- `optimize_rank_factors.py`: conservative monitored gradient optimizer using
  the shared model, fixed-gauge projection, line search, and retraction/refit.
- `optimize_rank_factors_preconditioned.py`: same optimizer shell with a
  per-block gradient-norm preconditioned direction. The useful mild setting from
  the comparison runs is `--block-power 0.5`.
- `optimize_rank_factors_lbfgs.py`: projected/retracted L-BFGS experiment using
  the mild block preconditioner as the initial inverse metric. Its default line
  search is now bracketed with an Armijo decrease check.
- `optimize_rank_factors_curl_trust.py`: curl-aware L-BFGS variant for the
  current formulation. It keeps the same objective and line search, but damps
  the localized hot `u1_Q -> u1_x2 -> curl` direction before normalization and
  records actual curl changes for each candidate.
- `optimize_rank_factors_damped_curl_loss.py`: vanilla gradient optimizer with
  the curl-equation loss multiplied pointwise by
  `(1 + x1^2 + x2^2)^(-p)`, useful for testing whether far-field curl weighting
  is the source of the huge vanilla gradient.
- `optimize_rank_factors_damped_curl_output.py`: sibling diagnostic that wires
  the same damping into the curl residual output before the loss sees it.
- `optimize_rank_factors_continuation.py`: staged constraint-weight continuation
  on top of the L-BFGS optimizer, carrying each stage's saved state into the
  next stage.
- `optimize_profile.py`: older low-dimensional correction-basis pilot, kept
  separate from the rank-factor optimizer.

All standalone Python entry points prepend a compatible `.python_deps` folder
before importing NumPy/SciPy when that folder exists, so optimizer and
diagnostic scripts use the same numerical libraries.

Run a specified number of rank-factor optimizer iterations with:

```powershell
cd C:\Users\Fan\Documents\Datafit
python optimize_rank_factors.py --max-iterations 50
```

The short form is:

```powershell
python optimize_rank_factors.py -n 50
```

Run the current direct L-BFGS baseline with:

```powershell
python optimize_rank_factors_lbfgs.py -n 50 --output-prefix compare_lbfgs_bracket_noreset_rank_optimization
```

Run the current continuation experiment with the same total accepted-step budget:

```powershell
python optimize_rank_factors_continuation.py --weights 1,3,10 --stage-iterations 10,10,30 --output-prefix compare_continuation_rank_optimization
```

Resume from the current best saved state at the final constraint weight with:

```powershell
python optimize_rank_factors_continuation.py --weights 10 --stage-iterations 25 --input-state extend10_continuation_w10_rank_optimization_stage01_w10_state.npz --output-prefix extend11_continuation_w10_rank_optimization
```

Run the current curl-aware same-formulation experiment from that saved state:

```powershell
python optimize_rank_factors_curl_trust.py -n 25 --input-state extend10_continuation_w10_rank_optimization_stage01_w10_state.npz --output-prefix compare_curl_trust_rank_optimization
```

Run the vanilla spatially damped curl-loss probe with:

```powershell
python optimize_rank_factors_damped_curl_loss.py -n 20 --curl-loss-power 0.0625 --output-prefix damped_curl_loss_p00625_rank_optimization
```

Use `--output-prefix some_name` to save `some_name_results.json`,
`some_name_history.csv`, and `some_name_state.npz` instead of overwriting the
default `rank_optimization_*` files.

Each accepted step prints one brief progress line with objective, objective
change, accepted step size, projected gradient norm, curl RMS, and maximum
gauge error.

Current optimizer comparison:

```text
direct bracketed L-BFGS, weight 10, 50 steps:  objective 10.91992175852
continuation 1 -> 3 -> 10, 50 total steps:    objective 10.91605231504
continuation + 250 more w10 steps, 300 total:  objective 10.90068021103
plain L-BFGS from best state, 25 steps:         objective 10.89995606905
curl-aware L-BFGS from best state, 25 steps:    objective 10.89873713824
```

The best saved state from these current-formulation runs is
`compare_curl_trust_rank_optimization_state.npz`. Starting from
`extend10_continuation_w10_rank_optimization_stage01_w10_state.npz`, the
curl-aware run drops the objective by `1.943072795783e-3` in 25 accepted steps,
versus `7.241419825590e-4` for plain same-start L-BFGS. The formal comparison
is in `compare_rank_optimization_curl_trust_vs_plain_extend10_summary.json`.
The hot `u1_Q` gradient channel remains after the run, so the method should be
understood as a targeted current-formulation preconditioner, not a structural
fix for the curl penalty stiffness.

Spatially damping the curl loss confirms the far-field diagnosis but does not
replace the current recommendation. With direct pointwise weight
`(1 + x1^2 + x2^2)^(-p)`, a 20-step vanilla sweep found `p=1/16` best among
`1/32, 1/16, 1/8, 1/4`: it reduced the original undamped objective to
`10.95392361796`, slightly better than vanilla's `10.95438481804`, and reduced
the first projected-gradient norm from about `4.30e14` to `1.05e13`. Stronger
damping ignored too much of the physical curl equation. The compact comparison
is in `compare_rank_optimization_damped_curl_loss_summary.json`.

Wiring the damping into the curl residual output instead gives only a very
weak sweet spot by the damped-output objective itself. In a 20-step sweep,
`p=1/256` reduced that objective by `0.419335%` from its initial value, versus
vanilla's `0.414683%`. Stronger output damping reduced the damped objective
less even when the original undamped objective sometimes improved. Extending
the sweep to `p=1/4, 1/2, 1` gave damped-output reductions of only
`0.001180%`, `0.000363%`, and `0.006243%`. The compact comparison is in
`compare_rank_optimization_damped_curl_output_summary.json`.

## Gradient Localization Diagnostic

To test whether the large gradient is global or localized, run:

```powershell
python diagnose_gradient_localization.py
```

This writes `gradient_localization_diagnostic_results.json` for the current best
state. The current result is strongly localized in coefficient space:

```text
total gradient norm: 3.954052e+12
u1_Q gradient block: 3.948461e+12, 99.717% of L2 gradient mass
u2_P gradient block: 2.080819e+11, 0.277% of L2 gradient mass
```

The largest grid-level field-adjoint components are also localized, mostly in
derivative terms and far-field boundary regions. For example, `omega.x2` has
90% of its L2 mass in 91 grid cells and almost all of it in the last 20 `x2`
columns; `zeta.x2` behaves similarly. The more ordinary value components are
much less concentrated.

Decomposing the rank-gradient formula by value, `x1`-derivative, and
`x2`-derivative contributions shows why the coefficient gradient lands in
`u1_Q`: the `u1_Q` block is essentially all from the `x2`-derivative term,
which is the `u1_x2` part of `curl = u1_x2 - u2_x1 - omega`.

```text
u1_Q total gradient norm: 3.948461e+12
  value contribution:        5.796464e+07
  x1-derivative contribution:7.592179e+08
  x2-derivative contribution:3.948461e+12
```

A projected-gradient step probe shows the overshoot is almost entirely in the
curl objective component. Steps from `3e-15` down to `1e-17` increase the
objective; at `1e-17`, the total objective change is `+9.475573e-06`, with curl
contributing `+9.477193e-06`. At `3e-18`, the objective decreases by
`-2.026889e-06` and the step is accepted.

To separate the global direction into variable blocks, run:

```powershell
python diagnose_block_step_overshoot.py
```

For the current curl-aware best state, `u1_Q` still carries `99.705%` of the
gradient L2 mass and accepts only a `3e-18` normalized block step before
overshooting at `1e-17`. The non-hot blocks can take much larger normalized
steps: for example `u1_P` accepts `3e-11`, `u1_s` accepts `3e-10`, `u2_s`
accepts `3e-9`, and scalar rates accept `1e-7` to `3e-7`. They eventually
overshoot too, but not at the tiny step scale set by `u1_Q`. The compact
summary is in `compare_block_gradient_step_overshoot_summary.json`.

## P/Q Row-Band Step Scaling

The rank-factor coefficient geometry is separable.  For every core, `P` rows
multiply the left B-spline basis (`core.x0`, `core.x1`) and are localized along
`x1`; `Q` rows multiply the right B-spline basis (`core.y0`, `core.y1`) and are
localized along `x2`.  For the velocity equations this means:

```text
divergence = u1x1 + u2x2
curl       = u1x2 - u2x1 - omega
```

So `u1_P` and `u2_Q` are mainly derivative-sensitive through divergence, while
`u1_Q` and `u2_P` are derivative-sensitive through curl.  The huge hot gradient
is still `u1_Q`, and it comes from the first `x2` boundary rows: the largest rows
are `u1_Q[3,*]` and `u1_Q[2,*]`, both supported at `x2 = 0 .. O(10^-2)` with
right-derivative basis maxima around `6e3`.

To map coefficient rows to grid bands and probe unit-norm row-band steps, run:

```powershell
python diagnose_pq_support_step_scaling.py
```

This writes `pq_support_step_scaling_diagnostic_results.json`; the compact
summary is `compare_pq_support_step_scaling_summary.json`.  The measured largest
accepted unit steps on the current curl-trust state are:

```text
u1_Q: x2<=0.02 -> 3e-18, 0.02<x2<=0.1 -> 1e-15,
      0.1<x2<=10 -> 1e-14, 10<x2<=100 -> 1e-13,
      100<x2<=1e4 -> 3e-12
u2_P: x1<=0.1 -> 1e-15, 0.1<x1<=1 -> 3e-15,
      1<x1<=100 -> 1e-14, 100<x1<=1e4 -> 3e-11
u1_P: x1<=0.02 -> 1e-16, 0.02<x1<=0.1 -> 1e-15,
      0.1<x1<=1 -> 1e-14, 1<x1<=100 -> 3e-14..3e-13,
      100<x1<=1e4 -> 1e-10, far tail -> 3e-12
u2_Q: x2<=0.02 -> 3e-17, 0.02<x2<=1 -> 3e-15,
      1<x2<=10 -> 1e-14, 10<x2<=100 -> 3e-13,
      100<x2<=1e4 -> 1e-10
```

A single power of `(1 + x1^2)` or `(1 + x2^2)` is not a good global model for
all four P/Q blocks.  `u2_P` is the cleanest case and roughly follows
`(1 + x1^2)^1.5` over the significant near/mid `x1` mass.  `u1_Q` is the
opposite: the boundary rows all have `basis_max_axis = 0`, so
`(1 + x2^2)^alpha` cannot distinguish the dangerous boundary band from nearby
rows; the formal fit gives an unusable large exponent.

The useful operational scaling is therefore banded.  Run:

```powershell
python probe_pq_scaled_directions.py
```

This reads the row-band safe steps and builds a P/Q-only direction by scaling
each gradient chunk by `largest_accepted_unit_step / ||gradient_chunk||` before
normalizing.  On the current curl-trust state:

```text
raw all-P/Q gradient:        accepted step 3e-18, best dJ = -2.040e-06
safe row-band P/Q gradient: accepted step 3e-09, best dJ = -8.950e-01
moderate power scaling:     accepted step 3e-13, best dJ = -6.469e-05
clipped fitted powers:      accepted step 3e-14, best dJ = -5.172e-03
```

The recommendation is to use the measured row-band scaling as the P/Q
preconditioner, not the literal global power-law fits.  If a formula is required
for implementation convenience, keep explicit boundary caps for `u1_Q` and
`u2_Q`, and use coordinate powers only as a secondary interpolation within
well-behaved bands.

To test whether that one-step result persists, run the saved-state optimizer
comparison:

```powershell
python optimize_rank_factors_pq_rowband.py --mode rowband_pq -n 30 `
  --state compare_curl_trust_rank_optimization_state.npz `
  --pq-diagnostic pq_support_step_scaling_diagnostic_results.json `
  --output-prefix compare_rowband_pq_30_from_curl_trust_rank_optimization

python optimize_rank_factors_pq_rowband.py --mode vanilla -n 30 `
  --state compare_curl_trust_rank_optimization_state.npz `
  --output-prefix compare_vanilla_30_from_curl_trust_rank_optimization
```

Both start from the same curl-trust state and use the unchanged objective.  The
30-step comparison is summarized in
`compare_pq_rowband_vs_vanilla_summary.json`:

```text
row-band P/Q: J 10.898737 ->  9.174090, dJ = -1.724647
vanilla:      J 10.898737 -> 10.898402, dJ = -0.000335
```

The row-band gain is mostly divergence loss:

```text
row-band P/Q component changes:
  divergence -1.692162, curl -0.032486, fomega/fzeta ~ unchanged
vanilla component changes:
  divergence -0.000001, curl -0.000334, fomega/fzeta ~ unchanged
```

So the row-band improvement is not just a one-shot artifact.  It continues to
reduce the loss for at least 30 accepted iterations from the current best state,
while vanilla and raw P/Q directions stay trapped at `O(1e-18)` steps.

### Constraint Weight Selection

The current vanilla optimizer default is the conservative fixed value
`constraint_weight = 3e-4` for divergence and curl.  Earlier experiments used
`constraint_weight = 10`; to test removing that old factor, run the same
saved-state comparison with
`--constraint-weight 1`:

```powershell
python optimize_rank_factors_pq_rowband.py --mode rowband_pq -n 30 `
  --constraint-weight 1 `
  --state compare_curl_trust_rank_optimization_state.npz `
  --pq-diagnostic pq_support_step_scaling_diagnostic_results.json `
  --output-prefix compare_rowband_pq_cw1_30_from_curl_trust_rank_optimization

python optimize_rank_factors_pq_rowband.py --mode vanilla -n 30 `
  --constraint-weight 1 `
  --state compare_curl_trust_rank_optimization_state.npz `
  --output-prefix compare_vanilla_cw1_30_from_curl_trust_rank_optimization
```

The compact summary is `compare_constraint_weight_1_vs_10_summary.json`.  In
weighted objective units, the total objective starts near `1.989874` instead of
`10.898737`, because divergence and curl are no longer multiplied by ten.  To
compare the four residual errors directly, divide the divergence/curl objective
components by the active constraint weight.  In those unweighted error units,
the 30-step row-band movement is:

```text
constraint_weight = 10:
  fomega +3.75e-08, fzeta -6.39e-08,
  divergence -1.69216e-01, curl -3.24863e-03

constraint_weight = 1:
  fomega +1.64e-08, fzeta -4.20e-07,
  divergence -1.67867e-01, curl -1.90095e-03
```

So removing the `10x` factor does not reverse the direction: row-band P/Q still
mostly reduces divergence, with a small curl reduction and negligible changes in
`fomega`/`fzeta`.  The main physical change is that curl improves less when the
curl equation is no longer given the `10x` weight.  Vanilla changes are tiny in
either weighting; in unweighted error units, vanilla's 30-step movement is
essentially the same for both weights:

```text
fomega ~0, fzeta ~0, divergence -1.25e-07, curl -3.34e-05
```

Reducing the constraint weight further shows when `Fomega` starts improving,
and also where the tradeoff becomes too expensive.  The extended sweep is
summarized in `compare_constraint_weight_extended_sweep_summary.json`.  All rows
below are 30-step row-band P/Q runs from the same curl-trust state; divergence
and curl are reported in unweighted error units:

```text
weight   d(Fomega+Fzeta)  dFomega       dFzeta        dDivergence   dCurl
10       -2.65e-08        +3.75e-08     -6.39e-08     -1.692e-01    -3.249e-03
1        -4.04e-07        +1.64e-08     -4.20e-07     -1.679e-01    -1.901e-03
0.1      -1.46e-06        -2.06e-07     -1.25e-06     -1.677e-01    -2.525e-03
0.01     -4.68e-06        -1.82e-06     -2.86e-06     -1.680e-01    -3.342e-03
0.001    -8.93e-06        -4.38e-06     -4.55e-06     -1.673e-01    -2.942e-03
0.0001   -4.07e-05        -2.40e-05     -1.67e-05     -1.735e-01    -3.249e-03
0.00001  -4.73e-04        -2.90e-04     -1.83e-04     +9.342e-01    +1.470e-02
0.000001 -3.67e-03        -2.27e-03     -1.40e-03     +8.324e+01    +1.486e+00
0.0000001 -3.72e-03       -2.27e-03     -1.45e-03     +1.122e+02    +5.553e+00
```

So yes: once the constraint weight is reduced below about `0.1`, the row-band
P/Q optimizer also decreases both `Fomega` and `Fzeta`.  The useful region seems
to be around `1e-4` if the constraints still matter physically: it makes the
`Fomega/Fzeta` decrease visible while divergence/curl still improve.  Below
that, the trend in `Fomega/Fzeta` continues, but the optimizer starts trading
away the constraint equations because they have become almost free in the
weighted objective.  At `1e-6` and `1e-7`, `Fomega+Fzeta` improves by about
`-3.7e-3`, but unweighted divergence/curl become dramatically worse.

A simple weight proxy that tracks this transition is the raw P/Q gradient
balance at unit constraint weight:

```text
rho = ||g_F,PQ|| / ||g_constraint,PQ||
g_F = gradient contribution of Fomega + Fzeta on u1/u2 P/Q
g_constraint = gradient contribution of divergence + curl on u1/u2 P/Q
```

This uses only quantities already available when forming the analytic gradient.
For the current curl-trust state:

```text
||g_F,PQ|| = 1.2718038e+07
||g_constraint,PQ|| = 3.9541734e+11
rho = 3.2163581e-05
```

Two simple proxies are useful:

```text
conservative: weight = clip(10*rho, 1e-4, 1e-2)  -> 3.22e-4 here
aggressive:   weight = clip(rho,    3e-5, 1e-2)  -> 3.22e-5 here
```

The conservative proxy is the safer default.  A nearby fixed test at `3e-4`
keeps all four unweighted components nonincreasing:

```text
weight 3e-4:
  d(Fomega+Fzeta) = -1.69e-05
  dDivergence     = -1.71e-01
  dCurl           = -3.23e-03
```

The aggressive proxy gives more F-equation progress, and the nearby `3e-5`
test still keeps all four components nonincreasing:

```text
weight 3e-5:
  d(Fomega+Fzeta) = -1.39e-04
  dDivergence     = -9.45e-02
  dCurl           = -3.16e-03
```

But it is close to the failure regime.  At `1e-5`, `Fomega+Fzeta` improves more
(`-4.73e-4`), but divergence and curl both get worse in unweighted units.  So
use the conservative proxy unless there is an explicit component-increase guard
that rejects steps when divergence or curl increase too much.

### Committed Vanilla Default-Weight Result

The repository includes one recorded vanilla run using the hard-coded default
`constraint_weight = 3e-4`:

```powershell
python optimize_rank_factors.py --max-iterations 30 `
  --output-prefix vanilla_cw0p0003_rank_optimization
```

The files are:

- `vanilla_cw0p0003_rank_optimization_results.json`
- `vanilla_cw0p0003_rank_optimization_history.csv`
- `vanilla_cw0p0003_rank_optimization_state.npz`

This is the plain projected-gradient optimizer, not one of the row-band or
L-BFGS experiments.  It accepted all 30 trial steps and reduced the weighted
objective from `1.0003000000000002` to `1.0002986303572519`
(`dJ = -1.3696427483e-6`).  The final residual RMS values were:

```text
fomega     7.9603408325e-08
fzeta      4.7097495742e-08
divergence 1.2299368046e-09
curl       7.6325487393e-06
```

## Streamfunction Velocity Diagnostic

The current optimizer still treats the fitted near-field velocity pieces
`u10f` and `u20f` as independent rank-factor cores. To test the proposed
streamfunction-derived velocity route, run:

```powershell
python diagnose_streamfunction_velocity.py
```

This writes `velocity_streamfunction_diagnostic_results.json`. The current
diagnostic shows that the saved streamfunction decomposition is consistent:

```text
relative RMS of psi - (psi0f + rat*Psi1): 3.5358242178e-06
```

However, directly differentiating `Vel.psi0f` with the Python B-spline
post-processing machinery is not an acceptable replacement for the current
velocity fit:

```text
current independent u-fit relative errors: u1=3.531855e-06, u2=3.537891e-06
full-mesh psi derivative total errors:     u1=2.294056e-03, u2=3.719304e-04
full-mesh psi derivative near errors:      u10f=1.108804e+00, u20f=8.997242e-01
best simple scaled-psi near-field error:   6.133622e-01
```

So the structural recommendation is still to derive velocity from a
streamfunction or vorticity operator, but not by simply replacing the `u1,u2`
cores with derivatives of `Vel.psi0f`. Trying simple product scalings of
`psi0f` before differentiating does not close the gap. The likely next step is
to recover or port the missing original `F_df`/streamfunction operator,
including its weighted far-field B-spline representation.

This agrees with `Numerics.pdf`, Appendix C.1: the profile fields use the
sixth-order B-spline representation (C.6), while the numerical streamfunction
uses a separate finite-element representation (C.8) with a larger streamfunction
mesh, far-field extrapolated bases, and the boundary weight
`rho_p(y) = arctan(1+y) - arctan(1)`. The current `data.mat` stores the resulting
`Vel.psi*` and velocity grids, but not the full finite-element operator or the
streamfunction coefficients needed to differentiate that representation exactly.

## Verification

The current Python run prints:

```text
omega fit RMS checks:  [4.7177956814e-11 1.2292858063e-09 5.8117823702e-10]
zeta fit RMS checks:   [5.5787647140e-11 2.3981468305e-10 7.8766363663e-10]
u1 relative RMS fit error:       3.5318552999e-06
u2 relative RMS fit error:       3.5378905867e-06
RMS u1x1 + u2x2:                 1.2299937836e-09
RMS u1x2 - u2x1 - omegafit:      7.6777215332e-06
Fomega: [2.2342190561e-06 5.9731056844e-07 6.1363658637e-08 7.9608629769e-08]
Fzeta:  [1.6670298256e-07 2.5025519440e-07 4.2303043205e-08 4.7103578979e-08]
```

The field fits, velocity relative errors, divergence check, and residual checks
match the MATLAB output at displayed precision. The vorticity identity check is
close but not bit-for-bit identical to MATLAB because the SVD and linear solves
in NumPy/SciPy do not choose exactly the same floating-point path as MATLAB.
