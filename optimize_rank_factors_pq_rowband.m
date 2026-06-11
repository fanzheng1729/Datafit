function output = optimize_rank_factors_pq_rowband(varargin)
%OPTIMIZE_RANK_FACTORS_PQ_ROWBAND MATLAB entry point for the row-band run.
%
% This is a thin wrapper around optimize_rank_factors.m so the MATLAB version
% reuses the existing model, gradient, retraction, line search, state saving,
% and JSON/CSV writers. Defaults mirror the current rat-enabled all-variable
% comparison: start from data.mat, use the all-variable diagnostic, run
% rowband_all at constraintWeight = 0.007. The row-band path now uses the
% direct scalar-gradient coordinate sweep for cl, cw, and rat while keeping
% the field blocks on measured row-band scaling.
%
% Examples:
%   optimize_rank_factors_pq_rowband()
%   optimize_rank_factors_pq_rowband(1, 'OutputPrefix', 'matlab_rowband_smoke')
%   optimize_rank_factors_pq_rowband('OutputPrefix', 'matlab_rowband_check')

defaults = {
    "Mode", "rowband_all", ...
    "StatePath", "", ...
    "RowbandDiagnostic", "all_variable_rowband_step_scaling_diagnostic_results.json", ...
    "MaxIterations", 30, ...
    "OutputPrefix", "matlab_from_begin_rat_w0p007_updated_diag_30", ...
    "ConstraintWeight", 0.007, ...
    "StepSweepInitialIterations", 5, ...
    "StepSweepPeriod", 5, ...
    "StepSweepMode", "neighbor"};

if ~isempty(varargin) && isnumeric(varargin{1})
    defaults = [defaults, {"MaxIterations", varargin{1}}]; %#ok<AGROW>
    varargin = varargin(2:end);
end

output = optimize_rank_factors(defaults{:}, varargin{:});
end
