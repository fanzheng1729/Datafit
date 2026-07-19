function output = optimize_rank_factors_pq_rowband(varargin)
%OPTIMIZE_RANK_FACTORS_PQ_ROWBAND MATLAB entry point for the row-band run.
%
% This is a thin wrapper around optimize_rank_factors.m so the MATLAB version
% reuses the existing model, gradient, retraction, line search, state saving,
% and JSON/CSV writers. Defaults mirror the current rat-enabled all-variable
% comparison: start from data.mat (overridable with DataPath), use the
% all-variable diagnostic, run
% rowband_all at constraintWeight = 0.007. By default the row-band path uses
% the direct scalar-gradient coordinate sweep for cl, cw, and rat, with two
% warmup sweeps and a 20-step scheduled neighbor-sweep period. Pass
% 'RowbandScalarUpdate','global_line_search' to use the previous shared global
% multiplier sweep instead.
%
% Examples:
%   optimize_rank_factors_pq_rowband()
%   optimize_rank_factors_pq_rowband(1, 'OutputPrefix', 'matlab_rowband_smoke')
%   optimize_rank_factors_pq_rowband('OutputPrefix', 'matlab_rowband_check')
%   optimize_rank_factors_pq_rowband(50, 'DataPath', 'data_stab.mat')

defaults = {
    "Mode", "rowband_all", ...
    "DataPath", "data.mat", ...
    "StatePath", "", ...
    "RowbandDiagnostic", "all_variable_rowband_step_scaling_diagnostic_results.json", ...
    "MaxIterations", 30, ...
    "OutputPrefix", "matlab_from_begin_rat_w0p007_updated_diag_30", ...
    "ConstraintWeight", 0.007, ...
    "RowbandScalarUpdate", "direct_gradient_coordinate_sweep", ...
    "StepSweepInitialIterations", 2, ...
    "StepSweepPeriod", 20, ...
    "StepSweepMode", "neighbor"};

if ~isempty(varargin) && isnumeric(varargin{1})
    defaults = [defaults, {"MaxIterations", varargin{1}}]; %#ok<AGROW>
    varargin = varargin(2:end);
end

output = optimize_rank_factors(defaults{:}, varargin{:});
end
