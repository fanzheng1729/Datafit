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
  preconditioned fixed-gauge projection in `rowband_all`.
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
- `--step-sweep-initial-iterations 5`
- `--step-sweep-period 5`
- `--step-sweep-mode neighbor`

Row-band modes use the measured natural direction norm with the standard
seven-point multiplier ladder.

The retained Python result artifacts aligned with this path are:

- `from_begin_initial_state.npz`
- `all_variable_rowband_step_scaling_diagnostic_results.json`
- `from_begin_rat_w0p007_updated_diag_30_results.json`
- `from_begin_rat_w0p007_updated_diag_30_history.csv`
- `from_begin_rat_w0p007_updated_diag_30_state.npz`

The updated diagnostic includes `rat`, uses constraint weight `0.007`, and
comes from `from_begin_initial_state.npz`. The recorded check reduces `J` from
`1.0070000000` to `0.1807256857`.

To regenerate the all-variable row-band diagnostic:

```powershell
python diagnose_pq_support_step_scaling.py
```
