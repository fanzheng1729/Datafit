# Datafit Profile Fit Bundle

This folder contains a compact MATLAB/Python bundle for fitting and checking a
saved approximate Boussinesq blowup profile. The notation is aligned with the
included papers:

- `Analysis.pdf`: Part I, the dynamic-rescaling equations and residual
  definitions.
- `Numerics.pdf`: Part II, the adaptive mesh, B-spline representation, far-field
  ansatz, and residual-estimation machinery.

This bundle is a runnable fit/check harness for the saved `data.mat` profile. It
does not reproduce the full rigorous interval-arithmetic verification from
`Numerics.pdf`.

## Paper Notation

The MAT file uses short names inherited from the original MATLAB computation.
The scripts now alias them to the paper notation:

| Data name | Paper name | Meaning |
| --- | --- | --- |
| `w` | `omega` | vorticity profile |
| `v` | `zeta` | `theta / x1`, not a velocity component |
| `wx1`, `wx2` | `omega_x1`, `omega_x2` | vorticity derivatives |
| `vx1`, `vx2` | `zeta_x1`, `zeta_x2` | `zeta` derivatives |
| `Vel.u1`, `Vel.u2` | `u1`, `u2` | velocity components |
| `cl`, `cw` | `c_l`, `c_omega` | dynamic-rescaling rates |

Because `theta = x1*zeta`, the source term in the vorticity equation is

```text
theta_x1 = zeta + x1*zeta_x1
```

The checked steady residuals are the dynamic-rescaling equations from
`Analysis.pdf` (2.10)-(2.11), rewritten in `omega,zeta` variables:

```text
Fomega = -(c_l*x + u).grad omega + c_omega*omega + theta_x1
Fzeta  = -(c_l*x + u).grad zeta + (2*c_omega - u1/x1)*zeta
```

The velocity identity check uses

```text
u1_x2 - u2_x1 = omega
```

## Files

- `runfit.m`: MATLAB entry point. It loads `data.mat`, aliases the saved data to
  paper notation, fits `omega`, `zeta`, `u1`, and `u2`, then prints residual
  checks.
- `optimize_rank_factors.m`: MATLAB translation of the explicit-`S`
  rank-factor optimizer. It uses the same optimized variables, analytic
  gradient, fixed-gauge projection, line search, and retraction/refit structure
  as the Python optimizer.
- `optimize_rank_factors_pq_rowband.m`: MATLAB wrapper for the current
  rat-enabled all-variable row-band optimizer setup. It reuses
  `optimize_rank_factors.m` and defaults to `constraintWeight = 0.007` with a
  neighbor scheduled step sweep.
- `runfit.py`: Python equivalent of the MATLAB fit/check path. See
  `README_PYTHON.md` for MAT-file loading and implementation details.
- `data.mat`: saved profile data, grids, velocity data, far-field coefficients,
  and B-spline matrices.
- `BS/`: minimal MATLAB B-spline helpers needed by `runfit.m`.
- `Cutoff_exp_modi.m`: helper required by function handles saved in `data.mat`.
- `Analysis.pdf`, `Numerics.pdf`: authoritative paper references for notation
  and numerical representation.

## Run MATLAB

Start MATLAB in this folder, or change into it before running:

```matlab
cd('C:\Users\Fan\Documents\Datafit')
run('runfit.m')
```

From PowerShell:

```powershell
matlab -batch "cd('C:\Users\Fan\Documents\Datafit'); run('runfit.m')"
```

`runfit.m` uses relative paths:

```matlab
addpath("BS\")
load data.mat
```

Run it from this folder unless those paths are changed.

## Run MATLAB Optimizer

The MATLAB optimizer is native MATLAB, not a wrapper around Python. From
PowerShell:

```powershell
matlab -batch "cd('C:\Users\Fan\Documents\Datafit'); optimize_rank_factors(20);"
```

or with explicit name-value options:

```powershell
matlab -batch "cd('C:\Users\Fan\Documents\Datafit'); optimize_rank_factors('MaxIterations',50,'OutputPrefix','matlab_rank_optimization');"
```

It prints one progress line after each accepted step:

```text
iter 01/20: J=... dJ=... step=... |g_T|=... curl=... gauge=...
```

By default it writes MATLAB-specific artifacts so it does not overwrite the
Python optimizer run:

```text
matlab_rank_optimization_results.json
matlab_rank_optimization_history.csv
matlab_rank_optimization_state.mat
```

The MATLAB path uses MATLAB's own SVD/QR and far-field function-handle
evaluation, so it is expected to be close to, but not bit-for-bit identical
with, the Python optimizer.

### MATLAB Row-Band Optimizer

The current row-band setup has a MATLAB wrapper:

```powershell
matlab -batch "cd('C:\Users\Fan\Documents\Datafit'); optimize_rank_factors_pq_rowband();"
```

Pass a leading integer to run a non-default number of iterations while keeping
the other row-band defaults:

```powershell
matlab -batch "cd('C:\Users\Fan\Documents\Datafit'); optimize_rank_factors_pq_rowband(2);"
```

The wrapper defaults to `Mode = rowband_all`, `ConstraintWeight = 0.007`, and
`MaxIterations = 30`. Row-band modes always start the line search from the
measured natural row-band direction norm and use the standard seven-point
multiplier ladder. The wrapper also uses `StepSweepInitialIterations = 5`,
`StepSweepPeriod = 5`, and `StepSweepMode = "neighbor"`. On scheduled sweep
iterations, the optimizer starts from the last accepted step multiplier, tests
the adjacent larger/smaller multipliers on the ladder, and keeps walking only
if the best accepted candidate is at the edge of the local bracket. On
intervening iterations it uses the last accepted multiplier directly. If that
single trusted step fails, `RecoveryStepSweep = true` falls back to one full
bracket sweep for that iteration. The wrapper starts from the canonical
`data.mat` fit and reads the rat-enabled all-variable diagnostic
`all_variable_rowband_step_scaling_diagnostic_results.json`.

To restore the old full-sweep behavior, set `StepSweepMode` to `"full"` and
`StepSweepPeriod` to `1`:

```powershell
matlab -batch "cd('C:\Users\Fan\Documents\Datafit'); optimize_rank_factors_pq_rowband(30,'StepSweepMode','full','StepSweepPeriod',1);"
```

Candidate line-search steps can be evaluated with MATLAB `parfor` when
Parallel Computing Toolbox is installed:

```powershell
matlab -batch "cd('C:\Users\Fan\Documents\Datafit'); optimize_rank_factors_pq_rowband(2,'UseParallel',true);"
```

`UseParallel` defaults to `false`. On the current 7-candidate line search,
serial evaluation was slightly faster than a warmed 4-worker process pool, so
parallel mode is mainly an optional experiment for heavier candidate workloads.
The neighbor scheduled-sweep policy is the recommended serial path. With the
updated weight-`0.007` diagnostic, the matching Python saved-state check
reduced `J` from `1.0070000000` to `0.1807256857` before the next line-search
bracket stopped.

For long MATLAB runs, enable periodic checkpointing:

```powershell
matlab -batch "cd('C:\Users\Fan\Documents\Datafit'); optimize_rank_factors_pq_rowband(200,'CheckpointPeriod',10,'OutputPrefix','matlab_long_rowband');"
```

This overwrites `matlab_long_rowband_checkpoint_state.mat`,
`matlab_long_rowband_checkpoint_history.csv`, and
`matlab_long_rowband_checkpoint_results.json` every 10 accepted steps. Resume
from the latest checkpoint by passing it as `StatePath`; for neighbor-sweep
runs, also pass the checkpoint JSON's `last_step_multiplier` as
`InitialStepMultiplier` so the local step ladder continues from the same
place:

```powershell
matlab -batch "cd('C:\Users\Fan\Documents\Datafit'); optimize_rank_factors_pq_rowband(200,'StatePath','matlab_long_rowband_checkpoint_state.mat','InitialStepMultiplier',0.01,'OutputPrefix','matlab_long_rowband_resume');"
```

## Run Python

Install dependencies, then run:

```powershell
python -m pip install -r requirements.txt
python runfit.py
```

For workspace-local dependencies:

```powershell
$cp = python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"
python -m pip install --target ".python_deps\$cp" -r requirements.txt
python runfit.py
```

`runfit.py` automatically adds a compatible local dependency folder to
`sys.path` when present. Binary packages such as NumPy/SciPy are tied to the
Python version, so Python 3.14 should use `.python_deps\cp314`, Python 3.12
should use `.python_deps\cp312`, and so on.

## Numerical Representation

The code follows the representation described in `Numerics.pdf`, Section 7 and
Appendix C:

- The approximate steady state is split into a semi-analytic far-field part and
  a faster-decaying numerical part.
- The numerical part is represented by sixth-order tensor-product B-splines on
  the adaptive mesh.
- The first coordinate uses odd or even reflected basis functions at `x1=0`,
  depending on the variable.
- Boundary and far-field extrapolation use the seventh-order coefficients listed
  in Appendix C.
- The stream-function far-field contribution `Psi1` is reconstructed from
  angular B-spline coefficients and the radial cutoff from Appendix D.1.3.

The fitting routine itself is a post-processing compression/check:

1. Compute an SVD of a two-dimensional field.
2. Keep modes whose weighted RMS contribution exceeds `epsSVD`.
3. Fit each retained one-dimensional singular vector with sixth-order
   B-splines.
4. Try coarsened remeshes with steps `1:10`, accepting a remesh when its
   singular-value-weighted error is below `epsfit`.
5. Reconstruct the field and derivatives from the fitted one-dimensional
   factors.

The derivatives are B-spline derivative evaluations, not finite differences of
the final fitted grid.

## Optimization Code

The current Python optimization side is kept as a compact mirror/check stack
for the rat-enabled all-variable row-band path:

- `rank_optimization_model.py` owns the explicit `P,Q,s,c_l,c_omega,rat`
  variables, objective evaluation, analytic gradients, fixed-gauge tangent
  projection, and retraction/refit hook.
- `rank_optimizer_helpers.py` contains the shared line-search, history, and
  NPZ persistence helpers used by Python optimizer entry points.
- `optimize_rank_factors_pq_rowband.py` runs the saved-state row-band optimizer.
  Its defaults are `rowband_all`, constraint weight `0.007`,
  `from_begin_initial_state.npz`, and
  `all_variable_rowband_step_scaling_diagnostic_results.json`.
- `diagnose_pq_support_step_scaling.py` regenerates row-band safe-step
  diagnostics. Its default target scope is now `all`, including `rat`.
- `gradient_check_rank.py`, `check_retraction_refit.py`, and
  `test_gradient_direction_signs.py` are retained as sanity checks for the
  shared model and chart mechanics.
- `rank_factor_tools.py`, `runfit.py`, and `local_deps.py` provide the reusable
  rank-chart, MATLAB-data, and local-dependency support.

The older vanilla-gradient, L-BFGS, curl-trust, damped-curl, P/Q-only result,
and profile-pilot Python files have been removed from the active workspace.

### Run the Python Row-Band Check

From PowerShell outside Codex:

```powershell
cd C:\Users\Fan\Documents\Datafit
$cp = python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"
python -m pip install --target ".python_deps\$cp" -r requirements.txt
python optimize_rank_factors_pq_rowband.py
```

By default, the optimizer writes:

```text
rowband_all_rank_optimization_results.json
rowband_all_rank_optimization_history.csv
rowband_all_rank_optimization_state.npz
```

## Verified Outputs

MATLAB currently prints these checks:

```text
omega fit RMS checks:  1.0e-08 * [0.0047 0.1229 0.0581]
zeta fit RMS checks:   1.0e-09 * [0.0558 0.2398 0.7877]

u1 relative RMS fit error:       3.5319e-06
u2 relative RMS fit error:       3.5379e-06
RMS u1_x1 + u2_x2:               1.2300e-09
RMS u1_x2 - u2_x1 - omegafit:    7.7040e-06

max original Fomega residual:    2.2342e-06
max fitted Fomega residual:      5.9731e-07
RMS original Fomega residual:    6.1364e-08
RMS fitted Fomega residual:      7.9609e-08

max original Fzeta residual:     1.6670e-07
max fitted Fzeta residual:       2.5026e-07
RMS original Fzeta residual:     4.2303e-08
RMS fitted Fzeta residual:       4.7104e-08
```

The Python implementation agrees with MATLAB on the field, velocity, divergence,
and residual checks at the displayed precision. Its vorticity identity check is
slightly different because NumPy/SciPy and MATLAB make slightly different
linear-algebra choices in the spline/SVD remesh path:

```text
MATLAB: 7.7040118598e-06
Python: 7.6777215332e-06
```
