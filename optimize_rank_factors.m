function output = optimize_rank_factors(varargin)
%OPTIMIZE_RANK_FACTORS Monitored explicit-S rank-factor optimizer in MATLAB.
%
% Examples:
%   optimize_rank_factors(20)
%   optimize_rank_factors('MaxIterations', 50)
%   optimize_rank_factors(2, 'OutputPrefix', 'matlab_smoke_rank_optimization')
%   optimize_rank_factors('Mode', 'rowband_all', 'StatePath', 'state.mat', ...
%       'RowbandDiagnostic', 'all_variable_rowband_step_scaling_diagnostic_results.json')

opts = parse_optimizer_options(varargin{:});
addpath("BS\")

fprintf("Monitored explicit-S rank-factor gradient optimization (MATLAB)\n");
fprintf("  max iterations:           %d\n", opts.maxIterations);
fprintf("  output prefix:            %s\n", opts.outputPrefix);
fprintf("  mode:                     %s\n", opts.mode);
if opts.mode == "rowband_all"
    fprintf("  rowband scalar update:    %s\n", opts.rowbandScalarUpdate);
end
fprintf("  constraint weight:        %.12g\n", opts.constraintWeight);
fprintf("  step sweep warmup:        %d\n", opts.stepSweepInitialIterations);
fprintf("  step sweep period:        %d\n", opts.stepSweepPeriod);
fprintf("  step sweep mode:          %s\n", opts.stepSweepMode);
fprintf("  step sweep start mult:    %.12g\n", opts.stepSweepStartMultiplier);
if isfinite(opts.initialStepMultiplier)
    fprintf("  initial step mult:        %.12g\n", opts.initialStepMultiplier);
end
fprintf("  recovery step sweep:      %d\n", opts.recoveryStepSweep);
fprintf("  checkpoint period:        %d\n", opts.checkpointPeriod);

model = build_model(opts.constraintWeight);
if strlength(opts.statePath) > 0
    fprintf("  starting state:           %s\n", opts.statePath);
    variables = load_optimizer_state(opts.statePath, model);
else
    variables = copy_variables(model.variables);
end
initial_variables = copy_variables(variables);
rowband_diagnostic = read_rowband_diagnostic(opts.rowbandDiagnostic, opts.mode);
target_blocks = target_blocks_for_mode(opts.mode, rowband_diagnostic);
if mode_needs_diagnostic(opts.mode)
    fprintf("  row-band diagnostic:      %s\n", opts.rowbandDiagnostic);
end

% The optimizer always evaluates the objective through the normalized
% residuals in model.scales. That keeps the four equations comparable even
% though the physical RMS sizes differ by many orders of magnitude.
current_evaluation = evaluate_model(model, variables);
[initial_objective, initial_components] = objective_and_components(model, current_evaluation.residuals);
current_objective = initial_objective;
current_components = initial_components;
initial_residuals = residual_rms_from_residuals(current_evaluation.residuals);
initial_gauge_errors = gauge_errors_from_fields(model, current_evaluation.fields);
step_scale = opts.initialStepScale;
last_step_multiplier = opts.initialStepMultiplier;
last_direct_multipliers = [];
history = struct([]);
candidate_history = struct([]);
stop_reason = "reached maximum iterations";
last_gradient_norm = NaN;
last_projected_gradient_norm = NaN;
last_directional_derivative = NaN;
paths = output_paths(opts.outputPrefix);
if opts.checkpointPeriod > 0
    fprintf("  checkpoint state:         %s\n", paths.checkpointState);
end

for iteration = 1:opts.maxIterations
    base_objective = current_objective;
    base_components = current_components;
    gradient = analytic_gradient(model, variables, current_evaluation);
    gradient_norm = variable_norm(gradient, model.variableKeys);
    last_gradient_norm = gradient_norm;

    if use_direct_rowband_scalar_update(opts)
        direct_gauges = gauge_gradients(model, variables);
        scheduled_sweep = should_sweep_steps(opts, iteration);
        search_step_scale = 1.0;
        if scheduled_sweep || isempty(last_direct_multipliers)
            if use_direct_neighbor_sweep(opts, last_direct_multipliers, iteration)
                [candidates, candidate_variables, candidate_evaluations] = rowband_direct_neighbor_candidates( ...
                    model, variables, gradient, rowband_diagnostic, target_blocks, ...
                    last_direct_multipliers, base_objective, base_components, opts, iteration, ...
                    "direct_sweep_neighbor", direct_gauges);
            else
                [field_multipliers, scalar_multipliers, direct_search_mode] = direct_sweep_multipliers( ...
                    opts, last_direct_multipliers, iteration, "direct_sweep");
                [candidates, candidate_variables, candidate_evaluations] = rowband_direct_coordinate_candidates( ...
                    model, variables, gradient, rowband_diagnostic, target_blocks, ...
                    field_multipliers, scalar_multipliers, ...
                    base_objective, base_components, opts, iteration, direct_search_mode, direct_gauges);
            end
        else
            [candidates, candidate_variables, candidate_evaluations] = rowband_direct_single_candidate( ...
                model, variables, gradient, rowband_diagnostic, target_blocks, ...
                last_direct_multipliers, base_objective, base_components, opts, iteration, direct_gauges);
        end
    else
        [raw_direction, preconditioner] = raw_direction_for_mode( ...
            opts.mode, variables, gradient, rowband_diagnostic, target_blocks, model);

        % Project the raw negative gradient into the tangent space of the origin
        % gauges before trying any step. This keeps c_l and c_omega tied to the
        % normalization conditions used by the dynamic-rescaling equations.
        tangent_direction = projected_tangent_direction( ...
            model, variables, gradient, raw_direction, preconditioner);
        [direction, projected_gradient_norm] = normalize_direction(tangent_direction, model.variableKeys);
        directional_derivative = variable_dot(gradient, direction, model.variableKeys);
        last_projected_gradient_norm = projected_gradient_norm;
        last_directional_derivative = directional_derivative;

        if ~isfinite(directional_derivative) || directional_derivative >= 0
            stop_reason = "projected direction was not a descent direction";
            break
        end

        search_step_scale = step_scale;
        if is_rowband_mode(opts.mode)
            % Row-band directions already carry measured safe group magnitudes.
            search_step_scale = projected_gradient_norm;
        end
        [step_multipliers, step_search_mode, scheduled_sweep] = step_multipliers_for_iteration( ...
            opts, iteration, last_step_multiplier);
        if step_search_mode == "neighbor_sweep"
            [candidates, candidate_variables, candidate_evaluations] = evaluate_neighbor_step_candidates( ...
                model, variables, direction, search_step_scale, step_multipliers, ...
                base_objective, base_components, opts, iteration);
        else
            [candidates, candidate_variables, candidate_evaluations] = evaluate_step_candidates( ...
                model, variables, direction, search_step_scale, step_multipliers, ...
                base_objective, base_components, opts, iteration, step_search_mode);
        end
    end
    iteration_trial_count = numel(candidates);
    candidate_history = append_struct_array(candidate_history, candidates);

    accepted = accepted_candidate_indices(candidates);
    if isempty(accepted) && ~scheduled_sweep && opts.recoveryStepSweep
        if use_direct_rowband_scalar_update(opts)
            [candidates, candidate_variables, candidate_evaluations] = rowband_direct_coordinate_candidates( ...
                model, variables, gradient, rowband_diagnostic, target_blocks, ...
                effective_direct_field_multipliers(), effective_direct_scalar_multipliers(), ...
                base_objective, base_components, opts, iteration, "direct_recovery_sweep", direct_gauges);
        else
            [candidates, candidate_variables, candidate_evaluations] = evaluate_step_candidates( ...
                model, variables, direction, search_step_scale, effective_step_multipliers(opts), ...
                base_objective, base_components, opts, iteration, "recovery_sweep");
        end
        iteration_trial_count = iteration_trial_count + numel(candidates);
        candidate_history = append_struct_array(candidate_history, candidates);
        accepted = accepted_candidate_indices(candidates);
    end
    if isempty(accepted)
        stop_reason = "no line-search candidate reduced the objective while preserving gauges";
        break
    end
    [~, local_index] = min([candidates(accepted).objective]);
    accepted_index = accepted(local_index);

    best = candidates(accepted_index);
    variables = candidate_variables{accepted_index};
    current_evaluation = candidate_evaluations{accepted_index};
    current_objective = best.objective;
    current_components = best.objective_components;
    step_scale = best.step;
    last_step_multiplier = best.step_multiplier;
    if use_direct_rowband_scalar_update(opts)
        last_direct_multipliers = direct_candidate_multipliers(best);
        projected_gradient_norm = best.step;
        directional_derivative = best.directional_derivative;
        last_projected_gradient_norm = projected_gradient_norm;
        last_directional_derivative = directional_derivative;
    end
    residuals = residual_rms_from_residuals(current_evaluation.residuals);
    gauge_errors = gauge_errors_from_fields(model, current_evaluation.fields);
    components = current_components;

    row = history_row( ...
        iteration, best.objective, best.objective_change, best.step, ...
        gradient_norm, projected_gradient_norm, directional_derivative, ...
        gauge_errors, residuals, opts.mode, search_step_scale, components, ...
        best.step_search_mode, best.step_multiplier, iteration_trial_count, variables);
    if use_direct_rowband_scalar_update(opts)
        row.field_step_multiplier = best.field_step_multiplier;
        row.cl_direct_multiplier = best.cl_direct_multiplier;
        row.cw_direct_multiplier = best.cw_direct_multiplier;
        row.rat_direct_multiplier = best.rat_direct_multiplier;
    end
    history = append_struct(history, row);

    fprintf("  iter %02d/%d: J=%.12e dJ=%.3e step=%.3e mult=%.3g search=%s trials=%d |d_raw|=%.3e div_loss=%.3e curl_loss=%.3e gauge=%.3e\n", ...
        iteration, opts.maxIterations, best.objective, best.objective_change, ...
        best.step, best.step_multiplier, best.step_search_mode, iteration_trial_count, ...
        projected_gradient_norm, components.divergence, components.curl, max_abs_struct(gauge_errors));

    if should_write_checkpoint(opts, numel(history))
        write_checkpoint( ...
            paths, variables, history, current_evaluation, ...
            current_objective, current_components, ...
            initial_objective, initial_components, initial_residuals, ...
            initial_gauge_errors, model, opts, target_blocks, iteration, ...
            step_scale, last_step_multiplier);
        fprintf("    checkpoint:             %s\n", paths.checkpointState);
    end

    if abs(best.objective_change) < opts.minObjectiveDecrease
        stop_reason = "accepted decrease fell below the minimum improvement";
        break
    end
end

final_objective = current_objective;
final_components = current_components;
final_residuals = residual_rms_from_residuals(current_evaluation.residuals);
final_gauge_errors = gauge_errors_from_fields(model, current_evaluation.fields);
accepted_count = numel(history);
if accepted_count > 0
    save(paths.state, "-struct", "variables", "-v7.3");
end

output = struct();
output.description = "Saved-state rank-factor optimizer comparison with optional row-band direction scaling (MATLAB).";
output.mode = opts.mode;
output.constraint_weight = opts.constraintWeight;
output.state_path = opts.statePath;
output.rowband_diagnostic_path = opts.rowbandDiagnostic;
output.accepted_steps = accepted_count;
if accepted_count > 0
    output.state_file = paths.state;
else
    output.state_file = "";
end
output.history_file = paths.history;
output.stop_reason = stop_reason;
output.objective_before = initial_objective;
output.objective_after = final_objective;
output.objective_change = final_objective - initial_objective;
output.relative_objective_change = (final_objective - initial_objective) / initial_objective;
output.objective_components_before = initial_components;
output.objective_components_after = final_components;
output.scalars_before = scalar_summary(initial_variables);
output.scalars_after = scalar_summary(variables);
output.initial_residual_rms = initial_residuals;
output.final_residual_rms = final_residuals;
output.gauge_targets = model.gaugeTargets;
output.gauge_errors_before = initial_gauge_errors;
output.gauge_errors_after = final_gauge_errors;
output.gauge_projection = gauge_projection_for_mode(opts.mode);
output.last_gradient_norm = last_gradient_norm;
output.last_projected_gradient_norm = last_projected_gradient_norm;
output.last_directional_derivative = last_directional_derivative;
output.line_search_step_multipliers = effective_step_multipliers(opts);
if opts.mode == "rowband_all"
    output.rowband_scalar_update = opts.rowbandScalarUpdate;
else
    output.rowband_scalar_update = "global_line_search";
end
if use_direct_rowband_scalar_update(opts)
    output.direct_field_multipliers = effective_direct_field_multipliers();
    output.direct_scalar_multipliers = effective_direct_scalar_multipliers();
    output.last_direct_multipliers = last_direct_multipliers;
else
    output.direct_field_multipliers = [];
    output.direct_scalar_multipliers = [];
    output.last_direct_multipliers = [];
end
output.initial_step_scale = opts.initialStepScale;
output.initial_step_multiplier = opts.initialStepMultiplier;
output.trusted_step_scale = step_scale;
output.last_step_multiplier = last_step_multiplier;
output.step_sweep_initial_iterations = opts.stepSweepInitialIterations;
output.step_sweep_period = opts.stepSweepPeriod;
output.step_sweep_mode = opts.stepSweepMode;
output.step_sweep_start_multiplier = opts.stepSweepStartMultiplier;
output.recovery_step_sweep = opts.recoveryStepSweep;
output.checkpoint_period = opts.checkpointPeriod;
output.checkpoint_state_file = checkpoint_file_field(opts, paths.checkpointState);
output.checkpoint_history_file = checkpoint_file_field(opts, paths.checkpointHistory);
output.checkpoint_results_file = checkpoint_file_field(opts, paths.checkpointResults);
output.max_iterations = opts.maxIterations;
output.min_objective_decrease = opts.minObjectiveDecrease;
output.gauge_tolerance = opts.gaugeTolerance;
output.target_blocks = cellstr(target_blocks);
output.ranks = ranks_struct(model);
output.history = history;
output.candidate_history = candidate_history;

write_history_csv(paths.history, history);
write_json(paths.results, output);

fprintf("  accepted steps:            %d\n", output.accepted_steps);
fprintf("  stop reason:               %s\n", output.stop_reason);
fprintf("  objective before:          %.12e\n", output.objective_before);
fprintf("  objective after:           %.12e\n", output.objective_after);
fprintf("  objective change:          %.12e\n", output.objective_change);
fprintf("  wrote history:             %s\n", paths.history);
if accepted_count > 0
    fprintf("  wrote state:               %s\n", paths.state);
end
fprintf("  wrote results:             %s\n", paths.results);
end

function opts = parse_optimizer_options(varargin)
% Parse name-value options while preserving the simple optimize_rank_factors(N)
% shorthand used for smoke tests.

opts = struct();
opts.maxIterations = 20;
opts.outputPrefix = "matlab_rank_optimization";
opts.mode = "vanilla";
opts.statePath = "";
opts.rowbandDiagnostic = "";
opts.constraintWeight = 10.0;
opts.minObjectiveDecrease = 1.0e-10;
opts.gaugeTolerance = 1.0e-8;
opts.initialStepScale = 3.0e-16;
opts.stepMultipliers = [10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01];
opts.stepSweepInitialIterations = 0;
opts.stepSweepPeriod = 1;
opts.stepSweepMode = "full";
opts.stepSweepStartMultiplier = 1.0;
opts.initialStepMultiplier = NaN;
opts.recoveryStepSweep = true;
opts.checkpointPeriod = 0;
opts.rowbandScalarUpdate = "direct_gradient_coordinate_sweep";

if ~isempty(varargin) && isnumeric(varargin{1})
    opts.maxIterations = varargin{1};
    varargin = varargin(2:end);
end

if mod(numel(varargin), 2) ~= 0
    error("Options must be name-value pairs.");
end

for k = 1:2:numel(varargin)
    name = lower(strrep(string(varargin{k}), "-", ""));
    value = varargin{k + 1};
    switch name
        case {"maxiterations", "n"}
            opts.maxIterations = value;
        case "outputprefix"
            opts.outputPrefix = string(value);
        case "mode"
            opts.mode = lower(string(value));
        case {"statepath", "state"}
            opts.statePath = string(value);
        case "rowbanddiagnostic"
            opts.rowbandDiagnostic = string(value);
        case "constraintweight"
            opts.constraintWeight = value;
        case "minobjectivedecrease"
            opts.minObjectiveDecrease = value;
        case "gaugetolerance"
            opts.gaugeTolerance = value;
        case "initialstepscale"
            opts.initialStepScale = value;
        case "stepmultipliers"
            opts.stepMultipliers = value;
        case {"stepsweepinitialiterations", "initialstepsweeps", "stepsweepwarmup"}
            opts.stepSweepInitialIterations = value;
        case {"stepsweepperiod", "stepsweepevery"}
            opts.stepSweepPeriod = value;
        case "stepsweepmode"
            opts.stepSweepMode = lower(string(value));
        case {"stepsweepstartmultiplier", "stepsweepstartmult"}
            opts.stepSweepStartMultiplier = value;
        case {"initialstepmultiplier", "initialstepmult", "laststepmultiplier", "laststepmult"}
            opts.initialStepMultiplier = value;
        case {"recoverystepsweep", "recoverstepsweep"}
            opts.recoveryStepSweep = logical_option(value);
        case {"checkpointperiod", "checkpointevery"}
            opts.checkpointPeriod = value;
        case {"rowbandscalarupdate", "scalarupdate", "scalarupdatemode", "scalarsweepmode"}
            opts.rowbandScalarUpdate = canonical_rowband_scalar_update(value);
        otherwise
            error("Unknown optimizer option: %s", varargin{k});
    end
end

if opts.maxIterations < 1 || opts.maxIterations ~= floor(opts.maxIterations)
    error("MaxIterations must be a positive integer.");
end
valid_modes = ["vanilla", "raw_all", "rowband_all"];
if ~any(opts.mode == valid_modes)
    error("Unknown optimizer mode: %s", opts.mode);
end
if opts.constraintWeight <= 0
    error("ConstraintWeight must be positive.");
end
if opts.stepSweepInitialIterations < 0 || opts.stepSweepInitialIterations ~= floor(opts.stepSweepInitialIterations)
    error("StepSweepInitialIterations must be a nonnegative integer.");
end
if opts.stepSweepPeriod < 1 || opts.stepSweepPeriod ~= floor(opts.stepSweepPeriod)
    error("StepSweepPeriod must be a positive integer.");
end
if ~any(opts.stepSweepMode == ["full", "neighbor"])
    error("StepSweepMode must be 'full' or 'neighbor'.");
end
if ~any(opts.rowbandScalarUpdate == ["global_line_search", "direct_gradient_coordinate_sweep"])
    error("RowbandScalarUpdate must be 'global_line_search' or 'direct_gradient_coordinate_sweep'.");
end
if opts.stepSweepStartMultiplier <= 0 || ~isfinite(opts.stepSweepStartMultiplier)
    error("StepSweepStartMultiplier must be positive and finite.");
end
if ~(isnumeric(opts.initialStepMultiplier) && isscalar(opts.initialStepMultiplier)) ...
        || (~isnan(opts.initialStepMultiplier) ...
        && (opts.initialStepMultiplier <= 0 || ~isfinite(opts.initialStepMultiplier)))
    error("InitialStepMultiplier must be NaN or a positive finite scalar.");
end
if opts.checkpointPeriod < 0 || opts.checkpointPeriod ~= floor(opts.checkpointPeriod)
    error("CheckpointPeriod must be a nonnegative integer.");
end
end

function paths = output_paths(prefix)
% Keep result, history, and restart state filenames mechanically tied to one
% prefix so a run can be archived without guessing which files belong together.

prefix = char(prefix);
paths = struct();
paths.results = sprintf("%s_results.json", prefix);
paths.history = sprintf("%s_history.csv", prefix);
paths.state = sprintf("%s_state.mat", prefix);
paths.checkpointResults = sprintf("%s_checkpoint_results.json", prefix);
paths.checkpointHistory = sprintf("%s_checkpoint_history.csv", prefix);
paths.checkpointState = sprintf("%s_checkpoint_state.mat", prefix);
end

function value = logical_option(raw)
% Accept MATLAB logicals, numeric 0/1, and simple strings in name-value args.

if islogical(raw)
    value = raw;
elseif isnumeric(raw)
    value = raw ~= 0;
else
    text = lower(string(raw));
    value = any(text == ["true", "1", "yes", "on"]);
end
end

function value = canonical_rowband_scalar_update(raw)
% Normalize aliases for the two row-band scalar update policies.

value = lower(string(raw));
switch value
    case {"global", "global_line_search", "line_search", "shared"}
        value = "global_line_search";
    case {"direct", "direct_gradient_coordinate_sweep", "coordinate", "separate"}
        value = "direct_gradient_coordinate_sweep";
    otherwise
        error("Unknown RowbandScalarUpdate: %s", raw);
end
end

function yes = use_direct_rowband_scalar_update(opts)
% True when rowband_all should sweep field/scalar multipliers separately.

yes = opts.mode == "rowband_all" ...
    && opts.rowbandScalarUpdate == "direct_gradient_coordinate_sweep";
end

function yes = use_direct_neighbor_sweep(opts, lastDirectMultipliers, iteration)
% Direct neighbor sweeps need a trusted multiplier tuple to start from.

yes = opts.stepSweepMode == "neighbor" ...
    && ~isempty(lastDirectMultipliers) ...
    && iteration > opts.stepSweepInitialIterations;
end

function multipliers = effective_step_multipliers(opts)
% The pushed row-band result uses the original seven-point bracket.

multipliers = opts.stepMultipliers;
end

function multipliers = effective_direct_field_multipliers()
% Field multipliers used by the direct-scalar row-band tuple search.

multipliers = [10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01, ...
    0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001];
end

function multipliers = effective_direct_scalar_multipliers()
% Absolute preconditioner scales for d_scalar = -m_scalar * g_scalar.

multipliers = [0.0, 1.0e-15, 3.0e-15, 1.0e-14, 3.0e-14, ...
    1.0e-13, 3.0e-13, 1.0e-12, 3.0e-12, 1.0e-11, 3.0e-11, ...
    1.0e-10, 3.0e-10, 1.0e-9, 3.0e-9, 1.0e-8, 3.0e-8, ...
    1.0e-7, 3.0e-7, 1.0e-6, 3.0e-6, 1.0e-5];
end

function [fieldMultipliers, scalarMultipliers, searchMode] = direct_sweep_multipliers(~, ~, ~, baseSearchMode)
% Full direct ladders used during warmup, explicit full mode, and recovery.

fieldMultipliers = effective_direct_field_multipliers();
scalarMultipliers = effective_direct_scalar_multipliers();
searchMode = baseSearchMode;
end

function multipliers = local_direct_multiplier_window(ladder, value)
% Return neighboring direct multipliers, with zero treated as a valid endpoint.

centerIndex = nearest_direct_multiplier_index(ladder, value);
indices = neighbor_window_indices(centerIndex, numel(ladder));
multipliers = ladder(indices);
end

function index = nearest_direct_multiplier_index(ladder, value)
% Direct scalar ladders include zero, so log-distance only applies when positive.

if value <= 0
    index = 1;
    return
end
positive = find(ladder > 0);
[~, localIndex] = min(abs(log(ladder(positive)) - log(value)));
index = positive(localIndex);
end

function steps = line_search_steps(stepScale, multipliers)
% Build the actual trial step sizes for the current trusted scale.

steps = stepScale * multipliers;
end

function yes = should_sweep_steps(opts, iteration)
% The default period of one preserves the original full sweep every iteration.

yes = opts.stepSweepPeriod == 1 ...
    || iteration <= opts.stepSweepInitialIterations ...
    || mod(iteration, opts.stepSweepPeriod) == 0;
end

function [multipliers, searchMode, scheduledSweep] = step_multipliers_for_iteration(opts, iteration, lastStepMultiplier)
% Choose either the full bracket or the last accepted bracket multiplier.

scheduledSweep = should_sweep_steps(opts, iteration);
if scheduledSweep
    if opts.stepSweepMode == "neighbor"
        multipliers = local_step_multiplier_window(opts, lastStepMultiplier);
        searchMode = "neighbor_sweep";
    else
        multipliers = effective_step_multipliers(opts);
        searchMode = "sweep";
    end
elseif isfinite(lastStepMultiplier) && lastStepMultiplier > 0
    multipliers = lastStepMultiplier;
    searchMode = "single";
else
    multipliers = 1.0;
    searchMode = "single_default";
end
end

function multipliers = local_step_multiplier_window(opts, lastStepMultiplier)
% First bracket for a neighbor sweep around the old accepted multiplier.

ladder = effective_step_multipliers(opts);
if isfinite(lastStepMultiplier) && lastStepMultiplier > 0
    center = lastStepMultiplier;
else
    center = opts.stepSweepStartMultiplier;
end
centerIndex = nearest_step_multiplier_index(ladder, center);
indices = neighbor_window_indices(centerIndex, numel(ladder));
multipliers = ladder(indices);
end

function index = nearest_step_multiplier_index(ladder, value)
% Compare in log scale because step multipliers are multiplicative.

[~, index] = min(abs(log(ladder) - log(value)));
end

function indices = neighbor_window_indices(centerIndex, ladderLength)
% Return old multiplier plus adjacent larger/smaller ladder entries.

lo = max(1, centerIndex - 1);
hi = min(ladderLength, centerIndex + 1);
indices = lo:hi;
end

function yes = is_rowband_mode(mode)
yes = startsWith(string(mode), "rowband_");
end

function modeName = gauge_projection_for_mode(mode)
% Row-band directions use the same measured metric for gauge projection.

if is_rowband_mode(mode)
    modeName = "preconditioned";
else
    modeName = "euclidean";
end
end

function yes = mode_needs_diagnostic(mode)
% raw_all needs the diagnostic for its target block list; rowband modes also
% need the measured safe step attached to each target block/group.

mode = string(mode);
yes = mode == "raw_all" || is_rowband_mode(mode);
end

function diagnostic = read_rowband_diagnostic(path, mode)
% Load the JSON emitted by diagnose_pq_support_step_scaling.py when a mode
% needs either its target list or measured row-band/block scales.

if ~mode_needs_diagnostic(mode)
    diagnostic = [];
    return
end
if strlength(string(path)) == 0
    error("%s mode requires RowbandDiagnostic.", mode);
end
diagnostic = jsondecode(fileread(path));
end

function blocks = target_blocks_for_mode(mode, diagnostic)
% Let all-variable modes follow the diagnostic that recorded which groups were
% actually probed.

mode = string(mode);
if mode == "raw_all" || mode == "rowband_all"
    blocks = target_blocks_from_diagnostic(diagnostic);
    if ~any(blocks == "rat")
        blocks = [blocks, "rat"];
    end
else
    blocks = strings(1, 0);
end
end

function blocks = target_blocks_from_diagnostic(diagnostic)
if isempty(diagnostic) || ~isfield(diagnostic, "target_blocks")
    error("Row-band diagnostic does not contain target_blocks.");
end
blocks = json_string_array(diagnostic.target_blocks);
end

function out = json_string_array(value)
% jsondecode has changed string-array behavior across MATLAB releases. This
% helper normalizes char/cell/string results into a row string array.

if iscell(value)
    out = strings(1, numel(value));
    for k = 1:numel(value)
        out(k) = string(value{k});
    end
elseif ischar(value)
    out = string(value);
else
    out = string(value);
    out = reshape(out, 1, []);
end
end

function variables = load_optimizer_state(path, model)
% Load a MATLAB state file with the same variable field names as the optimizer.
% This mirrors the Python saved-state entry point, but keeps MATLAB free of a
% runtime NumPy/NPZ dependency.

data = load(path);
variables = struct();
for keyName = model.variableKeys
    key = char(keyName);
    if ~isfield(data, key)
        if keyName == "rat"
            variables.(key) = model.variables.(key);
            continue
        end
        error("State file %s is missing variable %s.", path, key);
    end
    value = double(data.(key));
    targetSize = size(model.variables.(key));
    if ~isequal(size(value), targetSize)
        if numel(value) ~= numel(model.variables.(key))
            error("State variable %s has size [%s], expected [%s].", ...
                key, num2str(size(value)), num2str(targetSize));
        end
        value = reshape(value, targetSize);
    end
    variables.(key) = value;
end
end

function [direction, preconditioner] = raw_direction_for_mode(mode, variables, gradient, diagnostic, targetBlocks, model)
% Select the descent direction family requested by Mode.

mode = string(mode);
preconditioner = [];
switch mode
    case "vanilla"
        direction = scale_variables(gradient, -1.0, model.variableKeys);
    case "raw_all"
        direction = selected_raw_direction(variables, gradient, targetBlocks, model.variableKeys);
    case "rowband_all"
        preconditioner = rowband_preconditioner(gradient, diagnostic, targetBlocks, model.variableKeys);
        direction = scale_variables( ...
            apply_rowband_preconditioner(gradient, preconditioner, model.variableKeys), ...
            -1.0, model.variableKeys);
    otherwise
        error("Unknown optimizer mode: %s", mode);
end
end

function direction = selected_raw_direction(variables, gradient, targetBlocks, keys)
% Control direction: move only selected blocks, with the raw negative gradient.

direction = zero_like_variables(variables, keys);
for blockName = targetBlocks
    block = char(blockName);
    direction.(block) = -gradient.(block);
end
end

function preconditioner = rowband_preconditioner(gradient, diagnostic, targetBlocks, keys)
% Positive block/row scalings that define the measured row-band metric.

preconditioner = struct();
for keyName = keys
    key = char(keyName);
    preconditioner.(key) = zeros(size(gradient.(key)));
end
for blockName = targetBlocks
    block = char(blockName);
    blockGradient = gradient.(block);
    if ~isfield(diagnostic.block_results, block)
        if numel(blockGradient) == 1
            safeStep = fallback_scalar_safe_step(diagnostic);
            preconditioner.(block) = set_group_scale(preconditioner.(block), blockGradient, 1, safeStep);
            continue
        end
        error("Row-band diagnostic does not contain block %s.", block);
    end
    groups = diagnostic.block_results.(block).groups;
    for k = 1:numel(groups)
        group = groups(k);
        safeStep = group_safe_step(group);
        if isempty(safeStep) || safeStep <= 0
            continue
        end
        rows = (double(group.row_min) + 1):(double(group.row_max) + 1);
        preconditioner.(block) = set_group_scale(preconditioner.(block), blockGradient, rows, safeStep);
    end
end
end

function blockScale = set_group_scale(blockScale, blockGradient, rows, safeStep)
% Store safeStep / ||gradient group|| for one scalar, vector, or row band.

if numel(blockGradient) == 1
    normValue = abs(blockGradient);
    if normValue > 0
        blockScale = safeStep / normValue;
    end
elseif isvector(blockGradient)
    rows = rows(rows >= 1 & rows <= numel(blockGradient));
    chunk = blockGradient(rows);
    normValue = norm(chunk(:));
    if normValue > 0
        blockScale(rows) = safeStep / normValue;
    end
else
    rows = rows(rows >= 1 & rows <= size(blockGradient, 1));
    chunk = blockGradient(rows, :);
    normValue = norm(chunk(:));
    if normValue > 0
        blockScale(rows, :) = safeStep / normValue;
    end
end
end

function out = apply_rowband_preconditioner(variables, preconditioner, keys)
% Apply the current row-band metric to any variable-shaped dictionary.

out = zero_like_variables(variables, keys);
for keyName = keys
    key = char(keyName);
    out.(key) = preconditioner.(key) .* variables.(key);
end
end

function safeStep = fallback_scalar_safe_step(diagnostic)
% Reuse the safest measured scalar rate step for new scalar variables.

steps = [];
for name = ["cl", "cw"]
    key = char(name);
    if isfield(diagnostic.block_results, key)
        groups = diagnostic.block_results.(key).groups;
        for k = 1:numel(groups)
            step = group_safe_step(groups(k));
            if ~isempty(step) && step > 0
                steps(end + 1) = step; %#ok<AGROW>
            end
        end
    end
end
if isempty(steps)
    safeStep = 1.0e-7;
else
    safeStep = min(steps);
end
end

function preconditioner = rowband_direct_base_preconditioner(gradient, diagnostic, targetBlocks, keys)
% Row-band field metric with cl/cw/rat initially disabled.

preconditioner = rowband_preconditioner(gradient, diagnostic, targetBlocks, keys);
for name = ["cl", "cw", "rat"]
    key = char(name);
    if isfield(preconditioner, key)
        preconditioner.(key) = 0.0;
    end
end
end

function preconditioner = rowband_direct_preconditioner(basePreconditioner, multipliers, keys)
% Apply one field multiplier and direct scalar-gradient scales.

preconditioner = copy_variables(basePreconditioner);
for keyName = keys
    key = char(keyName);
    if ~isscalar(preconditioner.(key))
        preconditioner.(key) = multipliers.field * preconditioner.(key);
    end
end
preconditioner.cl = multipliers.cl;
preconditioner.cw = multipliers.cw;
preconditioner.rat = multipliers.rat;
end

function multipliers = direct_multiplier_tuple(fieldMultiplier, scalars)
% Normalize row-band direct multiplier tuples.

if nargin < 2 || isempty(scalars)
    scalars = struct();
end
multipliers = struct();
multipliers.field = double(fieldMultiplier);
multipliers.cl = scalar_field_or_zero(scalars, "cl");
multipliers.cw = scalar_field_or_zero(scalars, "cw");
multipliers.rat = scalar_field_or_zero(scalars, "rat");
end

function value = scalar_field_or_zero(values, name)
key = char(name);
if isfield(values, key)
    value = double(values.(key));
else
    value = 0.0;
end
end

function multipliers = direct_candidate_multipliers(candidate)
% Extract the direct multiplier tuple stored in a candidate summary.

multipliers = candidate.rowband_direct_multipliers;
end

function safeStep = group_safe_step(group)
% JSON null decodes to [] in MATLAB, which means "this group had no accepted
% measured unit step" and should be left at zero movement.

safeStep = group.trial_summary.largest_accepted_step;
if isempty(safeStep)
    return
end
safeStep = double(safeStep);
end

function model = build_model(constraintWeight)
% Build every fixed quantity needed by the reduced rank-factor objective.
%
% The dependent variables are omega, zeta, u1, and u2. Each is represented by
% P*S*Q' in a spline basis after applying the same damping/profile factors used
% by runfit.m. The PDE residuals then operate on reconstructed physical fields.

data = load("data.mat");
model = struct();
model.x1 = double(data.x1(:));
model.x2 = double(data.x2(:));
model.x1Col = model.x1;
model.x2Row = model.x2';
model.nGrid = numel(model.x1) * numel(model.x2);
model.constraintWeight = constraintWeight;

omega = double(data.w);
zeta = double(data.v);
u10f = double(data.Vel.u10f);
u20f = double(data.Vel.u20f);

model.profileFactor = sqrt(1 + model.x1Col.^2) ./ ...
    sqrt(1 + model.x1Col.^2 + model.x2Row.^2);
model.profileFactorX1 = model.profileFactor .* ...
    (model.x1Col .* model.x2Row.^2 ./ (1 + model.x1Col.^2) ./ ...
    (1 + model.x1Col.^2 + model.x2Row.^2));
model.profileFactorX2 = model.profileFactor .* ...
    (-model.x2Row ./ (1 + model.x1Col.^2 + model.x2Row.^2));

factor0u1 = 1 + model.x1Col.^2;
model.u1Factor = factor0u1 .^ 0.25;
model.u1FactorX1 = 0.5 * model.x1Col .* model.u1Factor ./ factor0u1;
factor0u2 = 1 + model.x2Row.^2;
model.u2Factor = factor0u2 .^ 0.25;
model.u2FactorX2 = 0.5 * model.x2Row .* model.u2Factor ./ factor0u2;

% Store one core per fitted field. Odd parity is used for omega, zeta, and u1;
% u2 is even at x1 = 0 and therefore uses BS6_interp2.
model.cores = [
    build_rank_core("omega", omega ./ model.profileFactor, model.x1, model.x2, 1, 1.0e-10)
    build_rank_core("zeta", zeta ./ model.profileFactor, model.x1, model.x2, 1, 1.0e-10)
    build_rank_core("u1", u10f ./ model.u1Factor, model.x1, model.x2, 1, 1.0e-11)
    build_rank_core("u2", u20f ./ model.u2Factor, model.x1, model.x2, 0, 1.0e-11)
];
model.coreNames = string({model.cores.name});

% The far-field stream-function piece is linear in the scalar coefficient rat.
% Store the unit-rat basis once and let the optimizer move rat with cl/cw.
model.farfieldVelocity = build_farfield_velocity_basis(data, model);
rat = double(data.rec(7));
cl = 4.0 * double(data.vx1(1, 1)) / double(data.wx1(1, 1));
cw = double(data.Vel.u1dx1(1, 1)) + cl / 2.0;

variables = struct();
variables.cl = cl;
variables.cw = cw;
variables.rat = rat;
for k = 1:numel(model.cores)
    core = model.cores(k);
    variables.(core.pKey) = core.p;
    variables.(core.qKey) = core.q;
    variables.(core.sKey) = core.s;
end
model.arrayKeys = build_array_keys(model);
model.variableKeys = [model.arrayKeys, "cl", "cw", "rat"];

% Start from a normalized chart. Retraction sorts singular values and makes
% the left/right factors orthonormal under the grid inner product.
model.variables = retract_variables(model, variables, true, 1.0e-10);

base = evaluate_model(model, model.variables);
model.gaugeTargets = gauge_values_from_fields(base.fields);
model.scales = struct();
% These scales define J = 1/2 mean((residual / initial_rms)^2). The constraint
% weight controls the relative pull of divergence and curl in this MATLAB
% reference optimizer.
model.scales.fomega = max(rms_all(base.residuals.fomega), realmin);
model.scales.fzeta = max(rms_all(base.residuals.fzeta), realmin);
model.scales.divergence = max(rms_all(base.residuals.divergence), realmin);
model.scales.curl = max(rms_all(base.residuals.curl), realmin);
end

function keys = build_array_keys(model)
% Flatten the P/Q/S field names into a stable order for vector-space helpers.

keys = strings(1, 3 * numel(model.cores));
index = 1;
for k = 1:numel(model.cores)
    core = model.cores(k);
    keys(index:index + 2) = [string(core.pKey), string(core.qKey), string(core.sKey)];
    index = index + 3;
end
end

function fixed = build_farfield_velocity_basis(data, model)
% Reconstruct the semi-analytic far-field velocity correction from the stored
% angular spline coefficients and polar derivative tables.

xycoe = XYcoef(double(data.gx1(:)), double(data.gx2(:)), double(data.alpha_b), data.Chi20, data.AG);
psi1 = Deri_Psi1(numel(data.gx1), numel(data.gx2), double(data.p_ag_coe), data.BS1d_large, 2, xycoe);
psi1 = Cell_2double(psi1);

n1 = numel(model.x1);
n2 = numel(model.x2);

fixed = struct();
fixed.u1 = -psi1(1:n1, 1:n2, 1, 2);
fixed.u1x1 = -psi1(1:n1, 1:n2, 2, 2);
fixed.u1x2 = -psi1(1:n1, 1:n2, 1, 3);
fixed.u2 = psi1(1:n1, 1:n2, 2, 1);
fixed.u2x1 = psi1(1:n1, 1:n2, 3, 1);
fixed.u2x2 = psi1(1:n1, 1:n2, 2, 2);
end

function core = build_rank_core(name, values, x1, x2, parity, epsSVD)
% Create the low-rank spline chart for one dependent field.
%
% values is the damped/scaled field to fit. The SVD decides the rank; the
% spline solves convert singular vectors into P and Q coefficient matrices.

[u, s_matrix, v] = svd(values, "econ");
s = diag(s_matrix);
rank = svd_order(u, s, v, epsSVD);
u = u(:, 1:rank);
v = v(:, 1:rank);
s = s(1:rank);
if parity == 1
    u(1, :) = 0;
end
mat = BS6mat(x1, x2, 2, parity);
if parity == 1
    % Odd fields vanish at the origin, so the first row would be a redundant
    % zero equation in the coefficient solve.
    p = mat{1, 1}(2:end, :) \ u(2:end, :);
else
    p = mat{1, 1} \ u;
end
q = mat{1, 2} \ v;
core = struct();
core.name = char(name);
core.parity = parity;
core.x0 = mat{1, 1};
core.x1 = mat{2, 1};
core.y0 = mat{1, 2};
core.y1 = mat{2, 2};
core.p = p;
core.q = q;
core.s = s(:);
core.rank = rank;
core.pKey = sprintf("%s_P", name);
core.qKey = sprintf("%s_Q", name);
core.sKey = sprintf("%s_s", name);
end

function n = svd_order(u, s, v, epsSVD)
% Keep singular vectors until the RMS-scaled rank-one contribution is below
% the requested tolerance.

n = numel(s);
for k = 1:n
    if rms_all(u(:, k)) * s(k) * rms_all(v(:, k)) <= epsSVD
        n = k;
        return
    end
end
end

function evaluation = evaluate_model(model, variables)
% Reconstruct physical fields, their first derivatives, and all equation
% residuals from the current rank-factor variables.

cache = struct();
for k = 1:numel(model.cores)
    core = model.cores(k);
    cache.(core.name) = core_eval(core, variables);
end

omega_core = cache.omega;
zeta_core = cache.zeta;
u1_core = cache.u1;
u2_core = cache.u2;

fields = struct();
% Undo the profile/damping factors and apply their product-rule derivative
% corrections. The velocity fields also receive the fixed far-field correction.
fields.omega = model.profileFactor .* omega_core.value;
fields.omega_x1 = model.profileFactor .* omega_core.x1 + model.profileFactorX1 .* omega_core.value;
fields.omega_x2 = model.profileFactor .* omega_core.x2 + model.profileFactorX2 .* omega_core.value;
fields.zeta = model.profileFactor .* zeta_core.value;
fields.zeta_x1 = model.profileFactor .* zeta_core.x1 + model.profileFactorX1 .* zeta_core.value;
fields.zeta_x2 = model.profileFactor .* zeta_core.x2 + model.profileFactorX2 .* zeta_core.value;
rat = variables.rat;
fields.u1 = model.u1Factor .* u1_core.value + rat * model.farfieldVelocity.u1;
fields.u1x1 = model.u1Factor .* u1_core.x1 + model.u1FactorX1 .* u1_core.value + rat * model.farfieldVelocity.u1x1;
fields.u1x2 = model.u1Factor .* u1_core.x2 + rat * model.farfieldVelocity.u1x2;
fields.u2 = model.u2Factor .* u2_core.value + rat * model.farfieldVelocity.u2;
fields.u2x1 = model.u2Factor .* u2_core.x1 + rat * model.farfieldVelocity.u2x1;
fields.u2x2 = model.u2Factor .* u2_core.x2 + model.u2FactorX2 .* u2_core.value + rat * model.farfieldVelocity.u2x2;

cl = variables.cl;
cw = variables.cw;
residuals = struct();
% The four residuals are Fomega, Fzeta, incompressibility, and vorticity-curl
% consistency. These are the four terms in the objective.
residuals.fomega = Fomega(cl, cw, model.x1Col, model.x2Row, fields.omega, fields.zeta, ...
    fields.omega_x1, fields.omega_x2, fields.u1, fields.u2, fields.zeta_x1);
residuals.fzeta = Fzeta(cl, cw, model.x1Col, model.x2Row, fields.zeta, fields.zeta_x1, ...
    fields.zeta_x2, fields.u1, fields.u2);
residuals.divergence = fields.u1x1 + fields.u2x2;
residuals.curl = fields.u1x2 - fields.u2x1 - fields.omega;

evaluation = struct();
evaluation.fields = fields;
evaluation.residuals = residuals;
evaluation.coreCache = cache;
end

function cache = core_eval(core, variables)
% Evaluate one P*S*Q' field and its x1/x2 derivatives in spline space.

p = variables.(core.pKey);
q = variables.(core.qKey);
s = variables.(core.sKey);
cache = struct();
cache.left = core.x0 * p;
cache.left_x1 = core.x1 * p;
cache.right = core.y0 * q;
cache.right_x2 = core.y1 * q;
cache.s = s(:);
[cache.value, cache.x1, cache.x2] = ...
    value_gradient(cache.left, cache.left_x1, cache.right, cache.right_x2, s);
end

function [value, value_x1, value_x2] = value_gradient(left, left_x1, right, right_x2, s)
% Matrix form of left*diag(s)*right' plus the two first derivatives.

weighted_left = left .* reshape(s, 1, []);
weighted_left_x1 = left_x1 .* reshape(s, 1, []);
value = weighted_left * right';
value_x1 = weighted_left_x1 * right';
value_x2 = weighted_left * right_x2';
end

function value = objective(model, variables)
evaluation = evaluate_model(model, variables);
value = objective_from_residuals(model, evaluation.residuals);
end

function value = objective_from_residuals(model, residuals)
% The objective is a normalized least-squares loss. Divergence and curl share
% the constraintWeight knob so their influence can be damped or emphasized.

[value, ~] = objective_and_components(model, residuals);
end

function components = objective_components(model, residuals)
% Return the four objective terms separately, matching the pushed JSON output.

components = struct();
components.fomega = 0.5 * mean((residuals.fomega ./ model.scales.fomega) .^ 2, "all");
components.fzeta = 0.5 * mean((residuals.fzeta ./ model.scales.fzeta) .^ 2, "all");
components.divergence = 0.5 * model.constraintWeight * ...
    mean((residuals.divergence ./ model.scales.divergence) .^ 2, "all");
components.curl = 0.5 * model.constraintWeight * ...
    mean((residuals.curl ./ model.scales.curl) .^ 2, "all");
end

function [value, components] = objective_and_components(model, residuals)
% Compute the four least-squares components once and sum them into J.

components = objective_components(model, residuals);
value = components.fomega + components.fzeta + components.divergence + components.curl;
end

function out = residual_rms_from_residuals(residuals)
out = struct();
out.fomega = rms_all(residuals.fomega);
out.fzeta = rms_all(residuals.fzeta);
out.divergence = rms_all(residuals.divergence);
out.curl = rms_all(residuals.curl);
end

function out = residual_rms(model, variables)
out = residual_rms_from_residuals(evaluate_model(model, variables).residuals);
end

function values = gauge_values_from_fields(fields)
% Origin gauges used to define the dynamic-rescaling rates.

values = struct();
values.omega_x1_00 = fields.omega_x1(1, 1);
values.theta_x1x1_00 = 2.0 * fields.zeta_x1(1, 1);
end

function values = gauge_values(model, variables)
values = gauge_values_from_fields(evaluate_model(model, variables).fields);
end

function errors = gauge_errors_from_fields(model, fields)
values = gauge_values_from_fields(fields);
errors = struct();
errors.omega_x1_00 = values.omega_x1_00 - model.gaugeTargets.omega_x1_00;
errors.theta_x1x1_00 = values.theta_x1x1_00 - model.gaugeTargets.theta_x1x1_00;
end

function errors = gauge_errors_for(model, variables)
errors = gauge_errors_from_fields(model, evaluate_model(model, variables).fields);
end

function lam = residual_lambdas(model, residuals)
% Derivative of the objective with respect to each residual array. These
% adjoint weights are propagated backward to field values and rank factors.

n = model.nGrid;
lam = struct();
lam.fomega = residuals.fomega ./ (model.scales.fomega ^ 2 * n);
lam.fzeta = residuals.fzeta ./ (model.scales.fzeta ^ 2 * n);
lam.divergence = model.constraintWeight * residuals.divergence ./ (model.scales.divergence ^ 2 * n);
lam.curl = model.constraintWeight * residuals.curl ./ (model.scales.curl ^ 2 * n);
end

function gradient = analytic_gradient(model, variables, evaluation)
% Analytic gradient of J with respect to every P/Q/S factor plus cl, cw, rat.

if nargin < 3
    evaluation = [];
end
fg = field_gradient(model, variables, evaluation);
evaluation = fg.evaluation;
fields = evaluation.fields;
lam = residual_lambdas(model, evaluation.residuals);
cl = variables.cl;

gradient = zero_like_variables(variables, model.variableKeys);
gradient.cl = sum(lam.fomega .* (-(model.x1Col .* fields.omega_x1 + model.x2Row .* fields.omega_x2)), "all") ...
    + sum(lam.fzeta .* (-(model.x1Col .* fields.zeta_x1 + model.x2Row .* fields.zeta_x2)), "all");
gradient.cw = sum(lam.fomega .* fields.omega, "all") + sum(lam.fzeta .* (2.0 * fields.zeta), "all");
gradient.rat = farfield_rat_gradient(model, fg);

% Push field adjoints through the damping factors and the spline/SVD chart.
gradient = add_rank_gradient(model, gradient, "omega", fg.omega.value, fg.omega.x1, fg.omega.x2, ...
    evaluation.coreCache.omega, model.profileFactor, model.profileFactorX1, model.profileFactorX2);
gradient = add_rank_gradient(model, gradient, "zeta", fg.zeta.value, fg.zeta.x1, fg.zeta.x2, ...
    evaluation.coreCache.zeta, model.profileFactor, model.profileFactorX1, model.profileFactorX2);
gradient = add_rank_gradient(model, gradient, "u1", fg.u1.value, fg.u1.x1, fg.u1.x2, ...
    evaluation.coreCache.u1, model.u1Factor, model.u1FactorX1, zeros(size(model.u1Factor)));
gradient = add_rank_gradient(model, gradient, "u2", fg.u2.value, fg.u2.x1, fg.u2.x2, ...
    evaluation.coreCache.u2, model.u2Factor, zeros(size(model.u2Factor)), model.u2FactorX2);
end

function value = farfield_rat_gradient(model, fg)
% Contract field adjoints with the unit-rat far-field velocity basis.

basis = model.farfieldVelocity;
value = sum(fg.u1.value .* basis.u1, "all") ...
    + sum(fg.u1.x1 .* basis.u1x1, "all") ...
    + sum(fg.u1.x2 .* basis.u1x2, "all") ...
    + sum(fg.u2.value .* basis.u2, "all") ...
    + sum(fg.u2.x1 .* basis.u2x1, "all") ...
    + sum(fg.u2.x2 .* basis.u2x2, "all");
end

function fg = field_gradient(model, variables, evaluation)
% Collect dJ/d(field), dJ/d(field_x1), and dJ/d(field_x2) for each physical
% dependent variable before applying profile factors and spline matrices.

if nargin < 3 || isempty(evaluation)
    evaluation = evaluate_model(model, variables);
end
fields = evaluation.fields;
lam = residual_lambdas(model, evaluation.residuals);
cl = variables.cl;
cw = variables.cw;
a1 = cl * model.x1Col + fields.u1;
a2 = cl * model.x2Row + fields.u2;
h_adjoint = zeros(size(fields.u1));
% The zeta residual contains u1/x1. At x1 = 0 the term is defined by the gauge
% limit and is handled as zero in this discrete residual.
h_adjoint(2:end, :) = (fields.zeta(2:end, :) .* lam.fzeta(2:end, :)) ./ model.x1(2:end);

fg = struct();
fg.evaluation = evaluation;
fg.omega = struct( ...
    "value", cw * lam.fomega - lam.curl, ...
    "x1", -a1 .* lam.fomega, ...
    "x2", -a2 .* lam.fomega);
fg.zeta = struct( ...
    "value", lam.fomega + (2.0 * cw - u1_over_x1(model, fields.u1)) .* lam.fzeta, ...
    "x1", model.x1Col .* lam.fomega - a1 .* lam.fzeta, ...
    "x2", -a2 .* lam.fzeta);
fg.u1 = struct( ...
    "value", -fields.omega_x1 .* lam.fomega - fields.zeta_x1 .* lam.fzeta - h_adjoint, ...
    "x1", lam.divergence, ...
    "x2", lam.curl);
fg.u2 = struct( ...
    "value", -fields.omega_x2 .* lam.fomega - fields.zeta_x2 .* lam.fzeta, ...
    "x1", -lam.curl, ...
    "x2", lam.divergence);
end

function out = u1_over_x1(model, u1)
% Safe discrete version of u1 / x1, leaving the origin row at zero.

out = zeros(size(u1));
out(2:end, :) = u1(2:end, :) ./ model.x1(2:end);
end

function gradient = add_rank_gradient(model, gradient, name, g_value, g_x1, g_x2, cache, factor, factor_x1, factor_x2)
% Chain the field adjoint through scaling factors, derivative matrices, and
% the bilinear P*S*Q' representation for one core.

core = get_core(model, name);
g_core = factor .* g_value + factor_x1 .* g_x1 + factor_x2 .* g_x2;
g_core_x1 = factor .* g_x1;
g_core_x2 = factor .* g_x2;
left = cache.left;
left_x1 = cache.left_x1;
right = cache.right;
right_x2 = cache.right_x2;
s = cache.s(:);

right_s = right .* reshape(s, 1, []);
right_x2_s = right_x2 .* reshape(s, 1, []);
left_s = left .* reshape(s, 1, []);
left_x1_s = left_x1 .* reshape(s, 1, []);

gradient.(core.pKey) = core.x0' * (g_core * right_s) ...
    + core.x1' * (g_core_x1 * right_s) ...
    + core.x0' * (g_core_x2 * right_x2_s);
gradient.(core.qKey) = core.y0' * (g_core' * left_s) ...
    + core.y0' * (g_core_x1' * left_x1_s) ...
    + core.y1' * (g_core_x2' * left_s);
gradient.(core.sKey) = sum(left .* (g_core * right), 1)' ...
    + sum(left_x1 .* (g_core_x1 * right), 1)' ...
    + sum(left .* (g_core_x2 * right_x2), 1)';
end

function core = get_core(model, name)
index = find(model.coreNames == string(name), 1);
core = model.cores(index);
end

function gauges = gauge_gradients(model, variables)
% Gradients of the two scalar gauge constraints used by the tangent projection.

zero = zero_like_variables(variables, model.variableKeys);
gauges = struct();
gauges.omega_x1_00 = single_gauge_gradient(model, "omega", variables, 1.0, zero);
gauges.theta_x1x1_00 = single_gauge_gradient(model, "zeta", variables, 2.0, zero);
end

function out = single_gauge_gradient(model, name, variables, scale, template)
% Gauge gradients only touch the origin row/column of the relevant core.

core = get_core(model, name);
cache = core_eval(core, variables);
factor = model.profileFactor(1, 1);
factor_x1 = model.profileFactorX1(1, 1);
left = cache.left(1, :);
left_x1 = cache.left_x1(1, :);
right = cache.right(1, :);
s = cache.s(:)';
out = copy_variables(template);
out.(core.pKey) = scale * (core.x1(1, :)' * (factor * right .* s) ...
    + core.x0(1, :)' * (factor_x1 * right .* s));
out.(core.qKey) = scale * (core.y0(1, :)' * (factor * left_x1 .* s + factor_x1 * left .* s));
out.(core.sKey) = scale * (factor * left_x1 .* right + factor_x1 * left .* right)';
end

function projected = projected_tangent_direction(model, variables, gradient, raw_direction, preconditioner)
% Remove components of raw_direction that would change the origin gauges to
% first order. The preconditioned mode uses the same positive row-band metric
% as the raw direction, preserving the first-order downhill guarantee.

if ~isempty(preconditioner)
    projected = preconditioned_projected_tangent_direction(model, variables, gradient, preconditioner);
    return
end

gauges = gauge_gradients(model, variables);
names = ["omega_x1_00", "theta_x1x1_00"];
gram = zeros(2, 2);
rhs = zeros(2, 1);
for i = 1:2
    rhs(i) = variable_dot(gauges.(names(i)), raw_direction, model.variableKeys);
    for j = 1:2
        gram(i, j) = variable_dot(gauges.(names(i)), gauges.(names(j)), model.variableKeys);
    end
end
if cond(gram) > 1.0e14
    correction = pinv(gram) * rhs;
else
    correction = gram \ rhs;
end

projected = copy_variables(raw_direction);
for i = 1:2
    gauge_gradient = gauges.(names(i));
    for key = model.variableKeys
        key = char(key);
        projected.(key) = projected.(key) - correction(i) * gauge_gradient.(key);
    end
end
end

function projected = preconditioned_projected_tangent_direction(model, variables, gradient, preconditioner, gauges)
% Build d = -B(g + A*lambda), with A' d = 0 and g' d <= 0.

if nargin < 5 || isempty(gauges)
    gauges = gauge_gradients(model, variables);
end
names = ["omega_x1_00", "theta_x1x1_00"];
preconditionedGradient = apply_rowband_preconditioner(gradient, preconditioner, model.variableKeys);
projected = scale_variables(preconditionedGradient, -1.0, model.variableKeys);

preconditionedGauges = struct();
for i = 1:2
    preconditionedGauges.(names(i)) = apply_rowband_preconditioner( ...
        gauges.(names(i)), preconditioner, model.variableKeys);
end

gram = zeros(2, 2);
rhs = zeros(2, 1);
for i = 1:2
    rhs(i) = variable_dot(gauges.(names(i)), projected, model.variableKeys);
    for j = 1:2
        gram(i, j) = variable_dot(gauges.(names(i)), preconditionedGauges.(names(j)), model.variableKeys);
    end
end
if cond(gram) > 1.0e14
    correction = pinv(gram) * rhs;
else
    correction = gram \ rhs;
end

for i = 1:2
    gauge_step = preconditionedGauges.(names(i));
    for key = model.variableKeys
        key = char(key);
        projected.(key) = projected.(key) - correction(i) * gauge_step.(key);
    end
end
end

function projected = direct_preconditioned_projected_tangent_direction(model, variables, gradient, basePreconditioner, multipliers, gauges)
% Direct scalar trials share raw gauge gradients but use multiplier-specific B.

if nargin < 6 || isempty(gauges)
    gauges = gauge_gradients(model, variables);
end
names = ["omega_x1_00", "theta_x1x1_00"];
preconditionedGradient = apply_direct_preconditioner(gradient, basePreconditioner, multipliers, model.variableKeys);
projected = scale_variables(preconditionedGradient, -1.0, model.variableKeys);

preconditionedGauges = struct();
for i = 1:2
    preconditionedGauges.(names(i)) = apply_direct_preconditioner( ...
        gauges.(names(i)), basePreconditioner, multipliers, model.variableKeys);
end

gram = zeros(2, 2);
rhs = zeros(2, 1);
for i = 1:2
    rhs(i) = variable_dot(gauges.(names(i)), projected, model.variableKeys);
    for j = 1:2
        gram(i, j) = variable_dot(gauges.(names(i)), preconditionedGauges.(names(j)), model.variableKeys);
    end
end
if cond(gram) > 1.0e14
    correction = pinv(gram) * rhs;
else
    correction = gram \ rhs;
end

for i = 1:2
    gauge_step = preconditionedGauges.(names(i));
    for key = model.variableKeys
        key = char(key);
        projected.(key) = projected.(key) - correction(i) * gauge_step.(key);
    end
end
end

function out = apply_direct_preconditioner(variables, basePreconditioner, multipliers, keys)
% Apply the direct row-band metric without rebuilding a full preconditioner.

out = zero_like_variables(variables, keys);
for keyName = keys
    key = char(keyName);
    switch key
        case 'cl'
            out.(key) = multipliers.cl .* variables.(key);
        case 'cw'
            out.(key) = multipliers.cw .* variables.(key);
        case 'rat'
            out.(key) = multipliers.rat .* variables.(key);
        otherwise
            out.(key) = (multipliers.field * basePreconditioner.(key)) .* variables.(key);
    end
end
end

function variables = retract_variables(model, variables, force, tolerance)
% Move every core back to the normalized P/S/Q chart after a trial step.

variables = copy_variables(variables);
for k = 1:numel(model.cores)
    core = model.cores(k);
    if ~force && core_chart_is_normalized(core, variables, tolerance)
        continue
    end
    left = core.x0 * variables.(core.pKey);
    right = core.y0 * variables.(core.qKey);
    s = variables.(core.sKey);
    result = retract_and_refit(left, right, s, core.x0, core.y0, core.parity);
    variables.(core.pKey) = result.leftCoefficients;
    variables.(core.qKey) = result.rightCoefficients;
    variables.(core.sKey) = result.singularValues;
end
end

function ok = core_chart_is_normalized(core, variables, tolerance)
% Decide whether a core is already close enough to the orthonormal chart.

errors = core_chart_errors(core, variables);
ok = errors.left_gram_error <= tolerance ...
    && errors.right_gram_error <= tolerance ...
    && errors.singular_sort_violation <= tolerance ...
    && errors.singular_negative_violation <= tolerance;
end

function errors = core_chart_errors(core, variables)
% Diagnostics for the chart constraints: left/right orthonormality and sorted
% nonnegative singular values.

left = core.x0 * variables.(core.pKey);
right = core.y0 * variables.(core.qKey);
s = variables.(core.sKey);
sorted_violation = 0;
if numel(s) > 1
    sorted_violation = max(0, max(s(2:end) - s(1:end - 1)));
end
negative_violation = max(0, -min(s));
errors = struct();
errors.left_gram_error = weighted_orthonormality_error(left);
errors.right_gram_error = weighted_orthonormality_error(right);
errors.singular_sort_violation = sorted_violation;
errors.singular_negative_violation = negative_violation;
end

function result = retract_and_refit(left, right, s, xBasis, yBasis, parity)
% Retract in value space, then refit the normalized factors back to spline
% coefficients. Odd x-parity again skips the origin row in the solve.

[new_left, new_right, new_s] = retract_factors(left, right, s);
if parity == 1
    p = xBasis(2:end, :) \ new_left(2:end, :);
else
    p = xBasis \ new_left;
end
q = yBasis \ new_right;
result = struct();
result.leftCoefficients = p;
result.rightCoefficients = q;
result.singularValues = new_s(:);
end

function [new_left, new_right, new_s] = retract_factors(left, right, s)
% QR-compress the two factor spaces, SVD the small core, and expand back.

[q_left, r_left] = qr(left, 0);
[q_right, r_right] = qr(right, 0);
core = r_left * diag(s(:)) * r_right';
[core_left, s_matrix, core_right] = svd(core, "econ");
new_left = q_left * core_left;
new_right = q_right * core_right;
new_s = diag(s_matrix);
[new_left, new_right] = fix_right_factor_signs(new_left, new_right);
end

function [left, right] = fix_right_factor_signs(left, right)
% Fix SVD sign ambiguity by making the largest entry in each right factor
% positive. This keeps saved states stable across platforms.

for col = 1:size(right, 2)
    [~, pivot] = max(abs(right(:, col)));
    if right(pivot, col) < 0
        left(:, col) = -left(:, col);
        right(:, col) = -right(:, col);
    end
end
end

function value = weighted_orthonormality_error(factors)
% Grid weights are uniform here, so orthonormality is the ordinary Gram error.

gram = factors' * factors;
value = norm(gram - eye(size(gram)), "fro");
end

function [candidates, candidate_variables, candidate_evaluations] = rowband_direct_coordinate_candidates( ...
    model, variables, gradient, diagnostic, targetBlocks, fieldMultipliers, scalarMultipliers, ...
    baseObjective, baseComponents, opts, iteration, stepSearchMode, gauges)
% No-duplicate direct search: field-only seed, then cl/cw/rat in turn.

if nargin < 13 || isempty(gauges)
    gauges = gauge_gradients(model, variables);
end
basePreconditioner = rowband_direct_base_preconditioner( ...
    gradient, diagnostic, targetBlocks, model.variableKeys);
candidates = struct([]);
candidate_variables = cell(0, 1);
candidate_evaluations = cell(0, 1);

seedTuples = struct([]);
for k = 1:numel(fieldMultipliers)
    seedTuples = append_struct(seedTuples, direct_multiplier_tuple(fieldMultipliers(k)));
end
[newCandidates, newVariables, newEvaluations] = evaluate_direct_tuple_candidates( ...
    model, variables, gradient, basePreconditioner, seedTuples, ...
    baseObjective, baseComponents, opts, iteration, stepSearchMode, gauges);
candidates = append_struct_array(candidates, newCandidates);
candidate_variables = [candidate_variables; newVariables]; %#ok<AGROW>
candidate_evaluations = [candidate_evaluations; newEvaluations]; %#ok<AGROW>

accepted = accepted_candidate_indices(candidates);
if isempty(accepted)
    return
end
[~, localIndex] = min([candidates(accepted).objective]);
currentMultipliers = direct_candidate_multipliers(candidates(accepted(localIndex)));

for scalarName = ["cl", "cw", "rat"]
    key = char(scalarName);
    if isstruct(scalarMultipliers)
        currentScalarMultipliers = scalarMultipliers.(key);
    else
        currentScalarMultipliers = scalarMultipliers;
    end
    trialTuples = struct([]);
    for k = 1:numel(currentScalarMultipliers)
        trial = currentMultipliers;
        trial.(key) = currentScalarMultipliers(k);
        trialTuples = append_struct(trialTuples, trial);
    end
    [newCandidates, newVariables, newEvaluations] = evaluate_direct_tuple_candidates( ...
        model, variables, gradient, basePreconditioner, trialTuples, ...
        baseObjective, baseComponents, opts, iteration, stepSearchMode + "_" + scalarName, gauges);
    candidates = append_struct_array(candidates, newCandidates);
    candidate_variables = [candidate_variables; newVariables]; %#ok<AGROW>
    candidate_evaluations = [candidate_evaluations; newEvaluations]; %#ok<AGROW>
    accepted = accepted_candidate_indices(candidates);
    if ~isempty(accepted)
        [~, localIndex] = min([candidates(accepted).objective]);
        currentMultipliers = direct_candidate_multipliers(candidates(accepted(localIndex)));
    end
end
end

function [candidates, candidate_variables, candidate_evaluations] = rowband_direct_neighbor_candidates( ...
    model, variables, gradient, diagnostic, targetBlocks, lastMultipliers, ...
    baseObjective, baseComponents, opts, iteration, stepSearchMode, gauges)
% Seed from the last accepted multiplier tuple, then widen each coordinate.

if nargin < 12 || isempty(gauges)
    gauges = gauge_gradients(model, variables);
end
basePreconditioner = rowband_direct_base_preconditioner( ...
    gradient, diagnostic, targetBlocks, model.variableKeys);
candidates = struct([]);
candidate_variables = cell(0, 1);
candidate_evaluations = cell(0, 1);
currentMultipliers = direct_multiplier_tuple(lastMultipliers.field, lastMultipliers);

for coordinateName = ["field", "cl", "cw", "rat"]
    if coordinateName == "field"
        ladder = effective_direct_field_multipliers();
    else
        ladder = effective_direct_scalar_multipliers();
    end
    coordinateSearchMode = stepSearchMode + "_" + coordinateName;
    [newCandidates, newVariables, newEvaluations] = evaluate_direct_coordinate_neighbor_candidates( ...
        model, variables, gradient, basePreconditioner, currentMultipliers, ...
        coordinateName, ladder, baseObjective, baseComponents, opts, iteration, ...
        coordinateSearchMode, gauges);
    candidates = append_struct_array(candidates, newCandidates);
    candidate_variables = [candidate_variables; newVariables]; %#ok<AGROW>
    candidate_evaluations = [candidate_evaluations; newEvaluations]; %#ok<AGROW>

    accepted = accepted_candidate_indices(candidates);
    if ~isempty(accepted)
        [~, localIndex] = min([candidates(accepted).objective]);
        currentMultipliers = direct_candidate_multipliers(candidates(accepted(localIndex)));
    end
end
end

function [candidates, candidate_variables, candidate_evaluations] = evaluate_direct_coordinate_neighbor_candidates( ...
    model, variables, gradient, basePreconditioner, centerMultipliers, coordinateName, ladder, ...
    baseObjective, baseComponents, opts, iteration, stepSearchMode, gauges)
% Walk one direct multiplier ladder outward from a trusted coordinate value.

if nargin < 13 || isempty(gauges)
    gauges = gauge_gradients(model, variables);
end
coordinateName = string(coordinateName);
centerIndex = nearest_direct_multiplier_index(ladder, centerMultipliers.(char(coordinateName)));
currentIndices = neighbor_window_indices(centerIndex, numel(ladder));
evaluated = false(1, numel(ladder));
candidates = struct([]);
candidate_variables = cell(0, 1);
candidate_evaluations = cell(0, 1);

while true
    newIndices = currentIndices(~evaluated(currentIndices));
    for index = newIndices
        trial = centerMultipliers;
        trial.(char(coordinateName)) = ladder(index);
        [newCandidates, newVariables, newEvaluations] = evaluate_direct_tuple_candidates( ...
            model, variables, gradient, basePreconditioner, trial, ...
            baseObjective, baseComponents, opts, iteration, stepSearchMode, gauges);
        for k = 1:numel(newCandidates)
            newCandidates(k).direct_neighbor_coordinate = coordinateName;
            newCandidates(k).direct_neighbor_index = index;
        end
        candidates = append_struct_array(candidates, newCandidates);
        candidate_variables = [candidate_variables; newVariables]; %#ok<AGROW>
        candidate_evaluations = [candidate_evaluations; newEvaluations]; %#ok<AGROW>
        evaluated(index) = true;
    end

    if isempty(candidates)
        accepted = [];
        currentCandidates = struct([]);
    else
        currentMask = ismember([candidates.direct_neighbor_index], currentIndices);
        currentCandidates = candidates(currentMask);
        accepted = accepted_candidate_indices(currentCandidates);
    end

    if isempty(accepted)
        expandedIndices = max(1, currentIndices(1) - 1):min(numel(ladder), currentIndices(end) + 1);
        if isequal(expandedIndices, currentIndices)
            break
        end
        currentIndices = expandedIndices;
        continue
    end

    [~, localIndex] = min([currentCandidates(accepted).objective]);
    best = currentCandidates(accepted(localIndex));
    bestIndex = best.direct_neighbor_index;
    if bestIndex == currentIndices(1) && bestIndex > 1
        currentIndices = neighbor_window_indices(bestIndex, numel(ladder));
    elseif bestIndex == currentIndices(end) && bestIndex < numel(ladder)
        currentIndices = neighbor_window_indices(bestIndex, numel(ladder));
    else
        break
    end

    if all(evaluated(currentIndices))
        break
    end
end
end

function [candidates, candidate_variables, candidate_evaluations] = rowband_direct_single_candidate( ...
    model, variables, gradient, diagnostic, targetBlocks, multipliers, ...
    baseObjective, baseComponents, opts, iteration, gauges)
% Evaluate the trusted direct multiplier tuple once.

if nargin < 11 || isempty(gauges)
    gauges = gauge_gradients(model, variables);
end
basePreconditioner = rowband_direct_base_preconditioner( ...
    gradient, diagnostic, targetBlocks, model.variableKeys);
[candidates, candidate_variables, candidate_evaluations] = evaluate_direct_tuple_candidates( ...
    model, variables, gradient, basePreconditioner, multipliers, ...
    baseObjective, baseComponents, opts, iteration, "direct_single", gauges);
end

function [candidates, candidate_variables, candidate_evaluations] = evaluate_direct_tuple_candidates( ...
    model, variables, gradient, basePreconditioner, multiplierTuples, ...
    baseObjective, baseComponents, opts, iteration, stepSearchMode, gauges)
% Evaluate row-band direct tuples, skipping non-descent degeneracies.

if nargin < 11 || isempty(gauges)
    gauges = gauge_gradients(model, variables);
end
candidates = struct([]);
candidate_variables = cell(0, 1);
candidate_evaluations = cell(0, 1);
for k = 1:numel(multiplierTuples)
    [summary, candidate, candidate_evaluation, ok] = direct_candidate( ...
        model, variables, gradient, basePreconditioner, multiplierTuples(k), ...
        baseObjective, baseComponents, opts, iteration, stepSearchMode, gauges);
    if ~ok
        continue
    end
    candidates = append_struct(candidates, summary);
    candidate_variables{end + 1, 1} = candidate; %#ok<AGROW>
    candidate_evaluations{end + 1, 1} = candidate_evaluation; %#ok<AGROW>
end
end

function [summary, candidate, candidate_evaluation, ok] = direct_candidate( ...
    model, variables, gradient, basePreconditioner, multipliers, ...
    baseObjective, baseComponents, opts, iteration, stepSearchMode, gauges)
% Build one direct-scalar preconditioned candidate.

if nargin < 11 || isempty(gauges)
    gauges = gauge_gradients(model, variables);
end
tangent_direction = direct_preconditioned_projected_tangent_direction( ...
    model, variables, gradient, basePreconditioner, multipliers, gauges);
[direction, projected_gradient_norm] = normalize_direction(tangent_direction, model.variableKeys);
summary = struct();
candidate = struct();
candidate_evaluation = struct();
ok = false;
if projected_gradient_norm == 0
    return
end
directional_derivative = variable_dot(gradient, direction, model.variableKeys);
if ~isfinite(directional_derivative) || directional_derivative >= 0
    return
end

[summary, candidate, candidate_evaluation] = candidate_summary( ...
    model, variables, direction, projected_gradient_norm, baseObjective, baseComponents, opts);
summary.iteration = iteration;
summary.step_multiplier = multipliers.field;
summary.step_multiplier_index = nearest_step_multiplier_index(effective_direct_field_multipliers(), multipliers.field);
summary.step_search_mode = stepSearchMode;
summary.direct_neighbor_coordinate = "";
summary.direct_neighbor_index = NaN;
summary.rowband_direct_multipliers = multipliers;
summary.field_step_multiplier = multipliers.field;
summary.cl_direct_multiplier = multipliers.cl;
summary.cw_direct_multiplier = multipliers.cw;
summary.rat_direct_multiplier = multipliers.rat;
summary.directional_derivative = directional_derivative;
ok = true;
end

function indices = accepted_candidate_indices(candidates)
% Accepted candidate indices, robust to an empty struct array.

if isempty(candidates)
    indices = [];
else
    indices = find([candidates.accepted]);
end
end

function [candidates, candidate_variables, candidate_evaluations] = evaluate_neighbor_step_candidates( ...
    model, variables, direction, searchStepScale, initialMultipliers, ...
    baseObjective, baseComponents, opts, iteration)
% Walk the multiplier ladder locally until the best accepted candidate is interior.

ladder = effective_step_multipliers(opts);
initialIndices = zeros(size(initialMultipliers));
for k = 1:numel(initialMultipliers)
    initialIndices(k) = nearest_step_multiplier_index(ladder, initialMultipliers(k));
end

candidates = struct([]);
candidate_variables = cell(0, 1);
candidate_evaluations = cell(0, 1);
evaluated = false(1, numel(ladder));
currentIndices = unique(initialIndices, "stable");

while true
    newIndices = currentIndices(~evaluated(currentIndices));
    if ~isempty(newIndices)
        [newCandidates, newVariables, newEvaluations] = evaluate_step_candidates( ...
            model, variables, direction, searchStepScale, ladder(newIndices), ...
            baseObjective, baseComponents, opts, iteration, "neighbor_sweep", newIndices);
        candidates = append_struct_array(candidates, newCandidates);
        candidate_variables = [candidate_variables; newVariables]; %#ok<AGROW>
        candidate_evaluations = [candidate_evaluations; newEvaluations]; %#ok<AGROW>
        evaluated(newIndices) = true;
    end

    currentMask = ismember([candidates.step_multiplier_index], currentIndices);
    currentCandidates = candidates(currentMask);
    accepted = find([currentCandidates.accepted]);
    if isempty(accepted)
        expandedIndices = max(1, currentIndices(1) - 1):min(numel(ladder), currentIndices(end) + 1);
        if isequal(expandedIndices, currentIndices)
            break
        end
        currentIndices = expandedIndices;
        continue
    end

    [~, localIndex] = min([currentCandidates(accepted).objective]);
    best = currentCandidates(accepted(localIndex));
    bestIndex = best.step_multiplier_index;
    if bestIndex == currentIndices(1) && bestIndex > 1
        centerIndex = bestIndex;
    elseif bestIndex == currentIndices(end) && bestIndex < numel(ladder)
        centerIndex = bestIndex;
    else
        break
    end

    currentIndices = neighbor_window_indices(centerIndex, numel(ladder));
    if all(evaluated(currentIndices))
        break
    end
end
end

function [candidates, candidate_variables, candidate_evaluations] = evaluate_step_candidates( ...
    model, variables, direction, searchStepScale, stepMultipliers, ...
    baseObjective, baseComponents, opts, iteration, stepSearchMode, stepMultiplierIndices)
% Evaluate one line-search bracket or a single trusted step.

if nargin < 11 || isempty(stepMultiplierIndices)
    ladder = effective_step_multipliers(opts);
    stepMultiplierIndices = zeros(size(stepMultipliers));
    for k = 1:numel(stepMultipliers)
        stepMultiplierIndices(k) = nearest_step_multiplier_index(ladder, stepMultipliers(k));
    end
end
steps = line_search_steps(searchStepScale, stepMultipliers);
candidate_summaries = cell(numel(steps), 1);
candidate_variables = cell(numel(steps), 1);
candidate_evaluations = cell(numel(steps), 1);
for k = 1:numel(steps)
    [summary, candidate, candidate_evaluation] = candidate_summary( ...
        model, variables, direction, steps(k), baseObjective, baseComponents, opts);
    summary.iteration = iteration;
    summary.step_multiplier = stepMultipliers(k);
    summary.step_multiplier_index = stepMultiplierIndices(k);
    summary.step_search_mode = stepSearchMode;
    candidate_summaries{k} = summary;
    candidate_variables{k} = candidate;
    candidate_evaluations{k} = candidate_evaluation;
end
candidates = [candidate_summaries{:}];
end

function [summary, candidate, candidate_evaluation] = candidate_summary(model, base_variables, direction, step, base_objective, base_components, opts)
% Build one line-search candidate and record the fields needed for acceptance.

candidate = add_scaled_variables(base_variables, direction, step, model.variableKeys);
candidate = retract_variables(model, candidate, false, 1.0e-10);
candidate_evaluation = evaluate_model(model, candidate);
[candidate_objective, components] = objective_and_components(model, candidate_evaluation.residuals);
gauge_errors = gauge_errors_from_fields(model, candidate_evaluation.fields);
objective_change = candidate_objective - base_objective;
summary = struct();
summary.step = step;
summary.objective = candidate_objective;
summary.objective_change = objective_change;
summary.accepted = objective_change < -opts.minObjectiveDecrease ...
    && max_abs_struct(gauge_errors) <= opts.gaugeTolerance;
summary.gauge_errors = gauge_errors;
summary.max_abs_gauge_error = max_abs_struct(gauge_errors);
summary.objective_components = components;
summary.objective_component_changes = subtract_component_structs(components, base_components);
end

function row = history_row(iteration, value, objective_change, accepted_step, gradient_norm, projected_gradient_norm, directional_derivative, gauge_errors, residuals, mode, searchStepScale, components, stepSearchMode, stepMultiplier, trialCount, variables)
% CSV-friendly record for one accepted step.

row = struct();
row.iteration = iteration;
row.objective = value;
row.objective_change = objective_change;
row.cl = variables.cl;
row.cw = variables.cw;
row.rat = variables.rat;
row.cl_over_cw = variables.cl / variables.cw;
row.accepted_step = accepted_step;
row.gradient_norm = gradient_norm;
row.projected_gradient_norm = projected_gradient_norm;
row.directional_derivative = directional_derivative;
row.max_abs_gauge_error = max_abs_struct(gauge_errors);
row.omega_x1_gauge_error = gauge_errors.omega_x1_00;
row.theta_x1x1_gauge_error = gauge_errors.theta_x1x1_00;
row.fomega_rms = residuals.fomega;
row.fzeta_rms = residuals.fzeta;
row.divergence_rms = residuals.divergence;
row.curl_rms = residuals.curl;
row.mode = mode;
row.search_step_scale = searchStepScale;
row.step_search_mode = stepSearchMode;
row.accepted_step_multiplier = stepMultiplier;
row.line_search_trial_count = trialCount;
row.fomega_loss = components.fomega;
row.fzeta_loss = components.fzeta;
row.divergence_loss = components.divergence;
row.curl_loss = components.curl;
end

function out = scalar_summary(variables)
% Scalar coefficients tracked alongside the rank-factor arrays.

out = struct();
out.cl = variables.cl;
out.cw = variables.cw;
out.rat = variables.rat;
out.cl_over_cw = variables.cl / variables.cw;
end

function out = subtract_component_structs(left, right)
% Difference of the four named objective components for candidate diagnostics.

out = struct();
out.fomega = left.fomega - right.fomega;
out.fzeta = left.fzeta - right.fzeta;
out.divergence = left.divergence - right.divergence;
out.curl = left.curl - right.curl;
end

function variables = copy_variables(variables)
% MATLAB structs copy by value; this function documents intent.
end

function out = zero_like_variables(variables, keys)
% Allocate a variable-shaped struct of zeros.

out = struct();
for key = keys
    key = char(key);
    out.(key) = zeros(size(variables.(key)));
end
end

function out = scale_variables(variables, scale, keys)
% Scalar multiplication in the product variable space.

out = struct();
for key = keys
    key = char(key);
    out.(key) = scale * variables.(key);
end
end

function out = add_scaled_variables(base, direction, scale, keys)
% Affine update base + scale * direction.

out = copy_variables(base);
for key = keys
    key = char(key);
    out.(key) = out.(key) + scale * direction.(key);
end
end

function value = variable_dot(left, right, keys)
% Euclidean product over all array variables and scalar rates.

value = 0.0;
for key = keys
    key = char(key);
    value = value + sum(left.(key) .* right.(key), "all");
end
end

function value = variable_norm(variables, keys)
value = sqrt(max(variable_dot(variables, variables, keys), 0.0));
end

function [out, value] = normalize_direction(direction, keys)
% Normalize a direction while returning its original norm.

value = variable_norm(direction, keys);
out = copy_variables(direction);
if value == 0
    return
end
for key = keys
    key = char(key);
    out.(key) = out.(key) / value;
end
end

function value = max_abs_struct(values)
% Maximum absolute entry over all fields in a scalar-diagnostic struct.

names = fieldnames(values);
value = 0.0;
for k = 1:numel(names)
    value = max(value, max(abs(values.(names{k})), [], "all"));
end
end

function out = ranks_struct(model)
% Serialize the retained SVD rank of each fitted field.

out = struct();
for k = 1:numel(model.cores)
    out.(model.cores(k).name) = model.cores(k).rank;
end
end

function yes = should_write_checkpoint(opts, acceptedCount)
% Checkpoint only after accepted steps, and only when explicitly requested.

yes = opts.checkpointPeriod > 0 ...
    && acceptedCount > 0 ...
    && mod(acceptedCount, opts.checkpointPeriod) == 0;
end

function value = checkpoint_file_field(opts, path)
% Keep disabled checkpoint fields explicit but empty in the final JSON.

if opts.checkpointPeriod > 0
    value = path;
else
    value = "";
end
end

function write_checkpoint(paths, variables, history, current_evaluation, current_objective, current_components, initial_objective, initial_components, initial_residuals, initial_gauge_errors, model, opts, target_blocks, iteration, stepScale, lastStepMultiplier)
% Write a restart state plus lightweight partial results for long runs.

save(paths.checkpointState, "-struct", "variables", "-v7.3");
write_history_csv(paths.checkpointHistory, history);
checkpoint = checkpoint_output( ...
    paths, history, current_evaluation, current_objective, current_components, ...
    initial_objective, initial_components, initial_residuals, initial_gauge_errors, ...
    model, opts, target_blocks, iteration, stepScale, lastStepMultiplier);
write_json(paths.checkpointResults, checkpoint);
end

function output = checkpoint_output(paths, history, current_evaluation, current_objective, current_components, initial_objective, initial_components, initial_residuals, initial_gauge_errors, model, opts, target_blocks, iteration, stepScale, lastStepMultiplier)
% Build a compact checkpoint JSON.  Candidate history is intentionally omitted.

current_residuals = residual_rms_from_residuals(current_evaluation.residuals);
current_gauge_errors = gauge_errors_from_fields(model, current_evaluation.fields);
output = struct();
output.description = "Checkpoint for saved-state rank-factor optimizer comparison (MATLAB).";
output.output_prefix = opts.outputPrefix;
output.iteration = iteration;
output.accepted_steps = numel(history);
output.mode = opts.mode;
output.constraint_weight = opts.constraintWeight;
output.state_path = opts.statePath;
output.rowband_diagnostic_path = opts.rowbandDiagnostic;
output.state_file = paths.checkpointState;
output.history_file = paths.checkpointHistory;
output.objective_before = initial_objective;
output.objective_after = current_objective;
output.objective_change = current_objective - initial_objective;
output.relative_objective_change = (current_objective - initial_objective) / initial_objective;
output.objective_components_before = initial_components;
output.objective_components_after = current_components;
output.initial_residual_rms = initial_residuals;
output.final_residual_rms = current_residuals;
output.gauge_targets = model.gaugeTargets;
output.gauge_errors_before = initial_gauge_errors;
output.gauge_errors_after = current_gauge_errors;
output.gauge_projection = gauge_projection_for_mode(opts.mode);
output.line_search_step_multipliers = effective_step_multipliers(opts);
if opts.mode == "rowband_all"
    output.rowband_scalar_update = opts.rowbandScalarUpdate;
else
    output.rowband_scalar_update = "global_line_search";
end
if use_direct_rowband_scalar_update(opts)
    output.direct_field_multipliers = effective_direct_field_multipliers();
    output.direct_scalar_multipliers = effective_direct_scalar_multipliers();
else
    output.direct_field_multipliers = [];
    output.direct_scalar_multipliers = [];
end
output.initial_step_scale = opts.initialStepScale;
output.initial_step_multiplier = opts.initialStepMultiplier;
output.trusted_step_scale = stepScale;
output.last_step_multiplier = lastStepMultiplier;
output.step_sweep_initial_iterations = opts.stepSweepInitialIterations;
output.step_sweep_period = opts.stepSweepPeriod;
output.step_sweep_mode = opts.stepSweepMode;
output.step_sweep_start_multiplier = opts.stepSweepStartMultiplier;
output.recovery_step_sweep = opts.recoveryStepSweep;
output.checkpoint_period = opts.checkpointPeriod;
output.max_iterations = opts.maxIterations;
output.min_objective_decrease = opts.minObjectiveDecrease;
output.gauge_tolerance = opts.gaugeTolerance;
output.target_blocks = cellstr(target_blocks);
output.ranks = ranks_struct(model);
output.history = history;
end

function write_history_csv(path, history)
% Write an empty file for no accepted steps so downstream scripts can still
% distinguish "no history" from "run did not finish".

if isempty(history)
    fid = fopen(path, "w");
    fclose(fid);
    return
end
table_history = struct2table(history);
writetable(table_history, path);
end

function write_json(path, value)
% MATLAB's PrettyPrint option is version-dependent, so fall back gracefully.

try
    text = jsonencode(value, "PrettyPrint", true);
catch
    text = jsonencode(value);
end
fid = fopen(path, "w");
fprintf(fid, "%s\n", text);
fclose(fid);
end

function array = append_struct(array, item)
% Append one struct while supporting the initially empty struct([]) case.

if isempty(array)
    array = item;
else
    array(end + 1) = item;
end
end

function array = append_struct_array(array, items)
% Append a batch of candidate summaries.

if isempty(items)
    return
end
if isempty(array)
    array = items;
else
    array = [array, items]; %#ok<AGROW>
end
end

function value = rms_all(array)
% MATLAB helper matching Python's global RMS.

value = sqrt(mean(array .^ 2, "all"));
end

function res = Fomega(cl, cw, x1, x2Row, omega, zeta, omega_x1, omega_x2, u1, u2, zeta_x1)
% Steady omega equation residual with theta_x = zeta + x1*zeta_x1.

theta_x = zeta + x1 .* zeta_x1;
res = -(cl * x1 + u1) .* omega_x1 - (cl * x2Row + u2) .* omega_x2 + cw * omega + theta_x;
end

function res = Fzeta(cl, cw, x1, x2Row, zeta, zeta_x1, zeta_x2, u1, u2)
% Steady zeta equation residual after writing theta = x1*zeta.

u1dx1 = zeros(size(u1));
u1dx1(2:end, :) = u1(2:end, :) ./ x1(2:end);
res = -(cl * x1 + u1) .* zeta_x1 - (cl * x2Row + u2) .* zeta_x2 + (2 * cw - u1dx1) .* zeta;
end

function mat = BS6mat(BSmesh, valmesh, ind, parity)
% Wrapper around the odd/even BS6 interpolation builders with constant weights.

f0 = @(x) 0;
f1 = @(x) 1;
Fconst = {f1, f0, f0, f0, f0, f0;
          f1, f0, f0, f0, f0, f0; };
r = BSmesh(end) / BSmesh(end - 1);
if parity == 1
    mat = BS6_interp(BSmesh, BSmesh, valmesh, valmesh, Fconst, r, 1, 0, ind);
else
    mat = BS6_interp2(BSmesh, BSmesh, valmesh, valmesh, Fconst, r, 1, 0, ind);
end
end

function XYcoe = XYcoef(x1, x2, alpha, Chi20, AG)
% Build far-field Cartesian coefficient tables for the semi-analytic tail.

sr = (x1 .^ 2 + x2' .^ 2) .^ (1/2);
sb = atan(x2' ./ x1);
sr = reshape(sr, [], 1);
sb = reshape(sb, [], 1);
a = 10;
l1 = 50000;
a2 = 100000;
i1 = find(x1 > a - 1, 1) - 1;
H = x1(end) - x1(end - 1);
i2 = round((sqrt(x1(end) ^ 2 + x2(end) ^ 2) - x1(end)) / H) + 3;
xr = x1(i1:end);
xr = [xr; x1(end) + (1:i2)' * H];
ord = 3;
[Psi_rad, ~] = Dchi(xr, ord + 2, a, l1, a2, alpha, Chi20);
XYcoe = Deri_polar_AGcoe(sr, sb, a, ord, Psi_rad, AG);
end

function fg = Leibni_prod(f, g, x, k)
% d_x^k(f*g) by the Leibniz rule, using cell arrays of derivative handles.

fg = zeros(size(x));
for i = 0:k
    fg = fg + f{i + 1}(x) .* g{k - i + 1}(x) .* nchoosek(k, i);
end
end

function chi = Assemble_chi(chi1, chi2, x, k)
% Derivative of chi = chi1 + (1 - chi1) * chi2.

chi = chi1(x, k);
for i = 0:k
    if i == k
        chi = chi + nchoosek(k, i) * chi2(x, i) .* (1 - chi1(x, k - i));
    else
        chi = chi + nchoosek(k, i) * chi2(x, i) .* (-chi1(x, k - i));
    end
end
end

function [Psi_rad, Chi] = Dchi(r, ord, a1, lam1, a2, alpha, Chi20)
% Construct radial cutoff derivatives and derivatives of r^(2-alpha)*chi(r).

itl = ~isa(r, "double") || ~isa(alpha, "double");
Chi10 = cell(ord + 1, 1);
syms x;
f = simplify(x ^ 7 / (1 + x ^ 2) ^ (7 / 2));
for i = 1:ord + 1
    if i > 1
        f = simplify(diff(f, x, 1));
    end
    f1 = matlabFunction(f);
    Chi10{i} = @(x) (x >= 0) .* f1(x);
end
if itl
    a1 = intval(a1);
    lam1 = intval(lam1);
    a2 = intval(a2);
    alpha = intval(alpha);
end
chi1 = @(x, k) Chi10{k + 1}((x - a1) / sqrt(lam1)) * lam1 ^ (-k / 2);
chi2 = @(x, k) Chi20{k + 1}((x - a2) / (9 * a2)) * (9 * a2) ^ (-k);
Chi = cell(ord + 1, 1);
for k = 0:ord
    Chi{k + 1} = @(x) Assemble_chi(chi1, chi2, x, k);
end
Pow_fun = cell(ord + 1, 1);
Psi_rad = cell(ord + 1, 1);
for i = 0:ord
    if i == 0
        fac = 1;
    else
        fac = fac * (2 - alpha - i + 1);
    end
    Pow_fun{i + 1} = @(x) fac * x .^ (2 - alpha - i);
end
for i = 0:ord
    Psi_rad{i + 1} = @(x) Leibni_prod(Chi, Pow_fun, x, i);
end
if ord > 7
    Psi_rad = Psi_rad(1:8);
end
end

function coe = Deri_polar_AGcoe(r, b, a1, ord, Psi_rad, AG)
% Cartesian derivative coefficients for A(r)B(beta) on flattened grid points.

if isa(r, "double") == 0 || isa(b, "double") == 0
    itl = 1;
else
    itl = 0;
end
nr = length(r);
nb = length(b);
r = reshape(r, nr, 1);
b = reshape(b, nb, 1);
if nr ~= nb
    error("Different length");
end
if ord >= size(AG, 1)
    error("Exceed angular degree");
end
if itl
    r_power = intval(zeros(nr, ord + 1));
    for i = 0:ord
        r_power(:, i + 1) = r .^ i;
    end
else
    r_power = r .^ (0:ord);
end
lg = find(r > a1);
Dpsi_rad = cell(ord + 1, 1);
for k = 0:ord
    Dpsi_rad{k + 1} = Psi_rad{k + 1}(r(lg));
end
coe = cell(ord + 1, ord + 1, ord + 1);
for deg = 0:ord
    for i = 0:deg
        j = deg - i;
        for l = 0:deg
            m = 0;
            coe{i + 1, j + 1, l + 1} = New_itl(zeros(nr, 1), itl);
            for k = 0:deg - l
                ag = AG{i + 1, j + 1, k + 1, l + 1}(b(lg));
                if isscalar(ag)
                    ag = ag .* ones(length(lg), 1);
                end
                m = m + ag ./ r_power(lg, deg - k + 1) .* Dpsi_rad{k + 1};
            end
            coe{i + 1, j + 1, l + 1}(lg) = m;
        end
    end
end
end

function Psi1 = Deri_Psi1(n1, n2, g, BS_wg, ord, XYcoe)
% Assemble derivatives of the semi-analytic stream function from radial
% coefficients and angular spline derivatives.

ord = min(ord, 6);
Psi1 = cell(ord + 1, ord + 1);
dg = cell(ord + 1, 1);
for i = 0:ord
    dg{i + 1} = (BS_wg{i + 1} * g) * (-1) ^ i;
end
for deg = 0:ord
    for i = 0:deg
        j = deg - i;
        m = 0;
        for l = 1:1 + deg
            m = m + XYcoe{i + 1, j + 1, l} .* dg{l};
        end
        Psi1{i + 1, j + 1} = reshape(m, n1, n2);
    end
end
end

function D = Cell_2double(F)
% Convert a rectangular cell array of same-sized arrays into a 4-D double.

[s1, s2] = size(F);
[s3, s4] = size(F{1, 1});
D = NaN(s3, s4, s1, s2);
for i = 1:s1
    for j = 1:s2
        if ~isempty(F{i, j})
            D(:, :, i, j) = F{i, j};
        end
    end
end
end

function M = New_itl(M0, itl)
% Preserve interval-arithmetic compatibility in the original MATLAB routines.

if itl
    M = intval(M0);
else
    M = M0;
end
end
