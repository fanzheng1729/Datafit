function output = optimize_rank_factors_pq_rowband(varargin)
%OPTIMIZE_RANK_FACTORS_PQ_ROWBAND MATLAB entry point for the pushed row-band run.
%
% This is a thin wrapper around optimize_rank_factors.m so the MATLAB version
% reuses the existing model, gradient, retraction, line search, state saving,
% and JSON/CSV writers.  Defaults mirror the pushed all-variable comparison:
% rowband_all, the all-variable diagnostic, 30 accepted attempts from the saved
% curl-trust state, no extra shrink ladder, and constraintWeight = 0.007.
%
% Examples:
%   optimize_rank_factors_pq_rowband()
%   optimize_rank_factors_pq_rowband(1, 'OutputPrefix', 'matlab_rowband_smoke')
%   optimize_rank_factors_pq_rowband('Mode', 'rowband_pq', ...
%       'RowbandDiagnostic', 'pq_support_step_scaling_diagnostic_results.json')

defaults = {
    "Mode", "rowband_all", ...
    "StatePath", "compare_curl_trust_rank_optimization_state.mat", ...
    "RowbandDiagnostic", "all_variable_rowband_step_scaling_diagnostic_results.json", ...
    "MaxIterations", 30, ...
    "OutputPrefix", "matlab_compare_rowband_all_cw0p007_30_from_curl_trust_rank_optimization", ...
    "ConstraintWeight", 0.007, ...
    "NoExtraShrinks", true, ...
    "RowbandUseNaturalStep", true};

if ~isempty(varargin) && isnumeric(varargin{1})
    defaults = [defaults, {"MaxIterations", varargin{1}}]; %#ok<AGROW>
    varargin = varargin(2:end);
end

output = optimize_rank_factors(defaults{:}, varargin{:});
end
