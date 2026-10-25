function [config, status] = applySitePolicy(config, policyPath)
%APPLYSITEPOLICY Override the referral boundary of a deployment config with a per-site policy file.
%
% The policy comes from python/calibrate_site.py. Only the four ordinal grade boundaries (and, if present, the
% referral-head threshold) are replaced; the bundle-bound config on disk is never modified. With no policy path the
% frozen thresholds stay and the status says so. Sample numbers in a site policy are NOT a validation.
    arguments
        config (1,1) struct
        policyPath (1,1) string = ""
    end
    status = struct('calibration_status', "UNCALIBRATED: frozen thresholds; run python/calibrate_site.py on a labelled sample of this site", ...
        'site_policy_path', "", 'site_policy_sha256', "", 'site_name', "", 'site_referable_cases', NaN);
    if strlength(policyPath) == 0
        return
    end
    if ~isfile(policyPath)
        error('NetrAI:SitePolicyNotFound', 'Site policy not found: %s', policyPath);
    end
    policy = jsondecode(fileread(policyPath));
    if ~isfield(policy, 'thresholds') || ~isnumeric(policy.thresholds) || numel(policy.thresholds) ~= 4 || ...
            any(~isfinite(policy.thresholds)) || any(diff(double(policy.thresholds(:))) < 0)
        error('NetrAI:InvalidSitePolicy', 'A site policy needs four finite, non-decreasing ordinal thresholds.');
    end
    if ~isfield(policy, 'status') || string(policy.status) ~= "site_calibrated"
        error('NetrAI:InvalidSitePolicy', 'The policy file is not a site-calibrated policy (status ~= site_calibrated).');
    end
    % A cut-off belongs to the model it was fitted for: refuse unbound policies and policies of another model/config.
    if ~isfield(config, 'deployment_identity')
        error('NetrAI:SitePolicyModelMismatch', 'The loaded model has no deployment identity to check the site policy against.');
    end
    identity = config.deployment_identity;
    if ~isfield(policy, 'model_sha256') || ~isfield(policy, 'config_sha256')
        error('NetrAI:SitePolicyModelMismatch', ['The site policy does not name the model it was fitted for ' ...
            '(model_sha256/config_sha256); re-create it with python/calibrate_site.py fit.']);
    end
    if ~strcmpi(string(policy.model_sha256), string(identity.model_sha256)) || ...
            ~strcmpi(string(policy.config_sha256), string(identity.config_sha256))
        error('NetrAI:SitePolicyModelMismatch', ['The site policy was fitted for a different model (%s) than the one loaded (%s).'], ...
            extractBefore(string(policy.model_sha256) + "............", 13), extractBefore(string(identity.model_sha256) + "............", 13));
    end
    config.grade_thresholds = double(policy.thresholds(:)).';
    if isfield(policy, 'head_threshold') && isnumeric(policy.head_threshold) && isscalar(policy.head_threshold)
        if ~isfinite(policy.head_threshold) || policy.head_threshold <= 0 || policy.head_threshold >= 1
            error('NetrAI:InvalidSitePolicy', 'head_threshold must lie in (0,1).');
        end
        config.referable_head_threshold = double(policy.head_threshold);
    end
    status.calibration_status = "site-calibrated (not independently validated)";
    status.site_policy_path = policyPath;
    status.site_policy_sha256 = string(fileSHA256(policyPath));
    if isfield(policy, 'site')
        if isfield(policy.site, 'name'), status.site_name = string(policy.site.name); end
        if isfield(policy.site, 'referable'), status.site_referable_cases = double(policy.site.referable); end
    end
end
