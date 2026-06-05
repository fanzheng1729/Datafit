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
- `optimize_rank_factors_pq_rowband.m`: MATLAB wrapper for the pushed
  all-variable row-band optimizer setup. It reuses `optimize_rank_factors.m`
  and defaults to the recorded `constraintWeight = 0.007`, 30-step run with a
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

The pushed row-band setup has a MATLAB wrapper:

```powershell
matlab -batch "cd('C:\Users\Fan\Documents\Datafit'); optimize_rank_factors_pq_rowband();"
```

Pass a leading integer to run a non-default number of iterations while keeping
the other row-band defaults:

```powershell
matlab -batch "cd('C:\Users\Fan\Documents\Datafit'); optimize_rank_factors_pq_rowband(2);"
```

The wrapper defaults to `Mode = rowband_all`, `ConstraintWeight = 0.007`,
`MaxIterations = 30`, `NoExtraShrinks = true`, and
`RowbandUseNaturalStep = true`. It also uses `StepSweepInitialIterations = 5`,
`StepSweepPeriod = 5`, and `StepSweepMode = "neighbor"`. On scheduled sweep
iterations, the optimizer starts from the last accepted step multiplier, tests
the adjacent larger/smaller multipliers on the ladder, and keeps walking only
if the best accepted candidate is at the edge of the local bracket. On
intervening iterations it uses the last accepted multiplier directly. If that
single trusted step fails, `RecoveryStepSweep = true` falls back to one full
bracket sweep for that iteration. The wrapper reads the converted MATLAB
starting state `compare_curl_trust_rank_optimization_state.mat` and the
diagnostic `all_variable_rowband_step_scaling_diagnostic_results.json`.

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
The neighbor scheduled-sweep policy is the recommended serial path: in a
30-step probe it used 50 candidate evaluations instead of 90 for scheduled full
sweeps or 210 for full sweeps every iteration, accepted all 30 steps, and
reached the same `J = 0.4762753188` as the scheduled-full run.

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

Extending the neighbor walk below the original ladder with
`IncludeExtraShrinks = true` changed the later trajectory but did not improve
the 30-step result. The run switched from multiplier `0.01` to `0.003` at
iteration 25, used 55 candidate evaluations, and ended at
`J = 0.4763737743`. The smaller step was locally better at iteration 25, but
then produced smaller follow-up decreases than staying at `0.01`, so extra
shrinks remain an experiment rather than the default. A 60-step comparison
confirmed the same trend: the default ladder stayed at `0.01` and reached
`J = 0.4726382314`, while the extended ladder stayed at `0.003` after
iteration 25 and reached `J = 0.4745918786`.

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

The optimization-related Python code is now split by role:

- `rank_optimization_model.py` is the shared explicit-`S` rank-factor model. It
  owns the variables `P,Q,s,c_l,c_omega`, objective evaluation, analytic
  gradients, fixed-gauge tangent projection, and retraction/refit hook.
- `gradient_check_rank.py` checks those shared analytic gradients against
  finite differences, including the explicit singular-amplitude variables `s`.
- `rank_factor_tools.py` contains the reusable rank-chart mechanics:
  `L diag(s) M.T` synthesis, weighted QR/SVD retraction, and B-spline refit.
- `check_retraction_refit.py` verifies that retraction/refit preserves all four
  fitted cores under the current discrete grid/SVD norm.
- `optimize_rank_factors.py` is the current explicit-`S` optimizer probe. It
  iterates analytic gradient steps, projects them onto the fixed-gauge tangent
  space, performs a line search, retracts/refits candidates, and saves
  `rank_optimization_results.json` plus `rank_optimization_state.npz` if a step
  is accepted.
- `optimize_rank_factors_lbfgs.py` is the projected/retracted L-BFGS experiment
  using the mild block preconditioner as its initial inverse metric.
- `optimize_rank_factors_continuation.py` runs staged constraint-weight
  continuation on top of the L-BFGS optimizer and carries each saved stage state
  into the next stage.
- `optimize_rank_factors_curl_trust.py` is the current same-formulation
  recommendation. It keeps the projected/retracted L-BFGS model but damps the
  curl-hot velocity block before line search, addressing the localized
  step-size bottleneck without changing the equations.
- `optimize_rank_factors_damped_curl_loss.py` is a vanilla-gradient diagnostic
  that multiplies the curl-equation loss by
  `(1 + x1^2 + x2^2)^(-p)` to test whether far-field curl weighting causes the
  huge gradient.
- `optimize_rank_factors_damped_curl_output.py` is the analogous diagnostic
  with the damping wired into the curl residual output before the loss sees it.
- `optimize_profile.py` is the older low-dimensional correction-basis pilot. It
  remains useful for comparison, but it is not the full rank-factor optimizer.

All standalone Python entry points prepend a compatible `.python_deps` folder
before importing NumPy/SciPy when one exists. This keeps the optimizer and
diagnostics on the same linear-algebra stack while avoiding incompatible binary
wheels from another Python version.

The best saved Python optimizer state is now the curl-aware same-start L-BFGS
run in `compare_curl_trust_rank_optimization_state.npz`. It starts from
`extend10_continuation_w10_rank_optimization_stage01_w10_state.npz`, the staged
continuation run `1 -> 3 -> 10` followed by ten 25-step extensions at
constraint weight `10`, and then takes 25 additional accepted weight-10 steps.
It reduces the objective from `10.900680211033425` to `10.898737138237642`,
with curl RMS `7.589398181576124e-06` and max gauge error about `8.44e-15`.
The fair same-start plain L-BFGS comparison is
`compare_extend10_plain_lbfgs_rank_optimization_state.npz`, which ends at
`10.899956069050866`; the curl-aware objective drop is about `2.68x` larger.
The comparison summary is
`compare_rank_optimization_curl_trust_vs_plain_extend10_summary.json`.

The committed optimizer recommendation is the row-band P/Q direction with the
low constraint weight `3e-4`.  Starting from
`compare_curl_trust_rank_optimization_state.npz`, the recorded 30-step run
reduces the weighted objective from `1.000296931941260` to
`1.000227625747525` while keeping all four unweighted components
nonincreasing.

The damped-curl-loss diagnostic confirms the localization picture but is not
the main recommendation: among powers `1/32, 1/16, 1/8, 1/4`, `p=1/16` gave
the best 20-step vanilla result, reducing the original undamped objective to
`10.95392361796` versus vanilla's `10.95438481804`. Stronger damping mostly
removed the far-field curl equation from the loss. The summary is
`compare_rank_optimization_damped_curl_loss_summary.json`.

When the damping is wired into the curl residual output instead, the best
20-step tested value was much weaker, `p=1/256`: it reduced the damped-output
objective by `0.419335%` from start, only slightly above vanilla's `0.414683%`.
Testing stronger output damping (`p=1/4, 1/2, 1`) made this worse: the
damped-output objective reductions were only `0.001180%`, `0.000363%`, and
`0.006243%`, respectively.
The summary is `compare_rank_optimization_damped_curl_output_summary.json`.

### Run the Rank-Factor Optimizer

From PowerShell outside Codex:

```powershell
cd C:\Users\Fan\Documents\Datafit
$cp = python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"
python -m pip install --target ".python_deps\$cp" -r requirements.txt
python optimize_rank_factors.py --max-iterations 50
```

Use `-n` as a short form:

```powershell
python optimize_rank_factors.py -n 50
```

By default, the optimizer writes:

```text
rank_optimization_results.json
rank_optimization_history.csv
rank_optimization_state.npz
```

For a test run that does not overwrite the main optimizer artifacts, choose a
different output prefix:

```powershell
python optimize_rank_factors.py --max-iterations 2 --output-prefix smoke_rank_optimization
```

The optimizer prints one brief progress line after each accepted step, for
example:

```text
iter 01/50: J=1.096590437052e+01 dJ=-3.410e-02 step=3.000e-16 |g_T|=4.298e+14 curl=7.641e-06 gauge=0.000e+00
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
