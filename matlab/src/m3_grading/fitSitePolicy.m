function policy = fitSitePolicy(scores, grades, config, options)
%FITSITEPOLICY Fit a per-site referral cut-off from expected-grade scores of a labelled site sample (no model needed).
%
% MATLAB counterpart of python/calibrate_site.py (same rule): the cut is the highest score at which the sample's sensitivity for
% grade >= 2 reaches TargetSensitivity (default 0.95, rule R2, confirmed on a fresh test). The policy binds the model and config it
% was fitted for (config.deployment_identity) and keeps the four ordinal boundaries strictly increasing, with the grade >= 2 boundary
% exactly at the cut. Sample numbers are optimistic and are NOT a validation.
    arguments
        scores (:,1) double {mustBeFinite}
        grades (:,1) double {mustBeInteger, mustBeInRange(grades, 0, 4)}
        config (1,1) struct
        options.TargetSensitivity (1,1) double {mustBeInRange(options.TargetSensitivity, 0.5, 1)} = 0.95
        options.SiteName (1,1) string = "unnamed site"
        options.ModelVariant (1,1) string = "E_768"
    end
    if numel(scores) ~= numel(grades)
        error('NetrAI:SiteSampleMismatch', '%d scores but %d grades.', numel(scores), numel(grades));
    end
    if ~isfield(config, 'deployment_identity')
        error('NetrAI:SitePolicyModelMismatch', 'The config has no deployment identity to bind the site policy to.');
    end
    referable = grades >= 2;
    nPos = nnz(referable); nNeg = nnz(~referable);
    if nPos < 5 || nNeg < 5
        error('NetrAI:SiteSampleTooSmall', 'Too few cases: %d referable / %d non-referable (need at least 5 of each; 40+ referable advised).', nPos, nNeg);
    end
    warningsList = strings(0, 1);
    if nPos < 40
        warningsList(end+1) = sprintf('Only %d referable cases: the fitted boundary is unstable (advised: at least 40).', nPos);
    end
    if nNeg < 150
        warningsList(end+1) = sprintf('Only %d non-referable cases: specificity cannot be judged (advised: at least 150).', nNeg);
    end
    positives = sort(scores(referable));
    allowedMisses = floor((1 - options.TargetSensitivity) * nPos);
    cut = positives(allowedMisses + 1) - 1e-9;
    base = double(config.grade_thresholds(:)).';
    gap = 1e-6;
    t0 = min(base(1), cut - gap); t2 = max(base(3), cut + gap); t3 = max(base(4), t2 + gap);
    called = scores >= cut; calledBase = scores >= base(2);
    identity = config.deployment_identity;
    policy = struct('status', 'site_calibrated', 'model_variant', char(options.ModelVariant), ...
        'model_sha256', char(identity.model_sha256), 'config_sha256', char(identity.config_sha256), ...
        'thresholds', [t0 cut t2 t3], ...
        'site', struct('name', char(options.SiteName), 'created_utc', char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z''')), ...
            'target_sample_sensitivity', options.TargetSensitivity, 'images', numel(scores), 'referable', nPos, 'non_referable', nNeg, ...
            'fitted_boundary', cut, 'base_boundary', base(2), ...
            'sample_operating_point_fitted', struct('sensitivity', mean(called(referable)), 'specificity', mean(~called(~referable))), ...
            'sample_operating_point_base', struct('sensitivity', mean(calledBase(referable)), 'specificity', mean(~calledBase(~referable))), ...
            'warnings', {cellstr(warningsList)}, ...
            'validation', 'NONE: sample numbers are optimistic (fitted on the same cases). Re-check on a separate labelled sample before screening.'));
end
