# Python Port Notes

`runfit.py` is a Python port of the MATLAB fit/check workflow in `runfit.m`.
It loads the numeric parts of `data.mat`, rebuilds the MATLAB function-handle
pieces from source formulas, fits the retained rank factors, and prints the
same field, velocity, divergence, curl, and residual checks as the MATLAB path.

## Dependencies

```powershell
python -m pip install -r requirements.txt
```

or, without touching the active Python environment:

```powershell
$cp = python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"
python -m pip install --target ".python_deps\$cp" -r requirements.txt
```

The scripts prepend a compatible local dependency folder such as
`.python_deps\cp314` when one exists. This keeps NumPy/SciPy wheels matched to
the Python version that is running the script.

## Current Python Stack

The active Python optimization code mirrors the rat-enabled all-variable
row-band experiment:

- `rank_optimization_model.py`: objective, analytic gradients, gauges,
  retraction/refit, and variables `P,Q,s,c_l,c_omega,rat`.
- `rank_optimizer_helpers.py`: shared candidate evaluation, line-search,
  history, and state persistence helpers.
- `optimize_rank_factors_pq_rowband.py`: saved-state row-band optimizer with
  preconditioned fixed-gauge projection in `rowband_all`. It supports both the
  current direct scalar-coordinate sweep and the previous shared global
  line-search sweep.
- `diagnose_pq_support_step_scaling.py`: all-variable row-band safe-step
  diagnostic generator.
- `gradient_check_rank.py`, `check_retraction_refit.py`, and
  `test_gradient_direction_signs.py`: model/chart sanity checks.
- `rank_factor_tools.py`, `runfit.py`, and `local_deps.py`: shared numerical
  support.

Old standalone vanilla-gradient, L-BFGS, curl-trust, damped-curl, and
low-dimensional profile-pilot files have been removed from the active Python
side.

## Run

```powershell
python optimize_rank_factors_pq_rowband.py
```

Current defaults are:

- `--mode rowband_all`
- `--constraint-weight 0.007`
- `--max-iterations 30`
- `--state from_begin_initial_state.npz`
- `--rowband-diagnostic all_variable_rowband_step_scaling_diagnostic_results.json`
- `--rowband-scalar-update direct_gradient_coordinate_sweep`
- `--step-sweep-initial-iterations 2`
- `--step-sweep-period 20`
- `--step-sweep-mode neighbor`
- `--step-sweep-start-multiplier 1.0`

Global row-band mode uses the measured natural direction norm with the
standard seven-point multiplier ladder. Direct scalar mode uses separate field
and scalar multiplier ladders described below.

`--rowband-scalar-update direct_gradient_coordinate_sweep` is the current
default. It starts with two full direct coordinate sweeps:

```text
1. field-only multiplier sweep
2. cl multiplier sweep
3. cw multiplier sweep
4. rat multiplier sweep
```

After the two warmup sweeps, unscheduled iterations reuse the last accepted
direct tuple `(field, cl, cw, rat)`. On scheduled iterations 20, 40, 60, ...
the neighbor search starts from that tuple and adaptively widens each
coordinate window if no local candidate is accepted or the best accepted
candidate is on the edge.

Use the previous shared global line search with:

```powershell
python optimize_rank_factors_pq_rowband.py -n 30 --rowband-scalar-update global_line_search --output-prefix py_global_neighbor_check
```

In global mode, all variables share one projected row-band direction and one
step multiplier from:

```text
[10, 3, 1, 0.3, 0.1, 0.03, 0.01]
```

The updated diagnostic includes `rat`, uses constraint weight `0.007`, and
comes from `from_begin_initial_state.npz`.

By default, the Python optimizer writes:

```text
rowband_all_rank_optimization_results.json
rowband_all_rank_optimization_history.csv
rowband_all_rank_optimization_state.npz
```

Python writes final artifacts only. It does not currently write periodic
checkpoint files during the run, so use the MATLAB wrapper for unattended
overnight runs that need checkpoint monitoring and resumability. Use Python for
parity checks, shorter optimizer tests, and inspecting the same algorithm in a
more compact implementation.

Recent from-begin checks from `J ~= 1.007`:

```text
Direct default, 20 accepted steps:
  J = 1.0070000000 -> 0.1267376768
  iteration 20 used adaptive direct-neighbor search with 13 trials

Global neighbor, 30 accepted steps:
  J = 1.0070000000 -> 0.1242733102
  direct scalar ladders were empty, confirming the global path
```

To regenerate the all-variable row-band diagnostic:

```powershell
python diagnose_pq_support_step_scaling.py
```

For the checkpointed overnight MATLAB runbook, see `README.md`, section
`MATLAB Row-Band Optimizer`.
