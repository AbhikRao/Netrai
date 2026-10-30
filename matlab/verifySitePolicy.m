function summary = verifySitePolicy(imagePaths, grades, policyPath, outputPath, options)
%VERIFYSITEPOLICY Check a site policy on a SEPARATE labelled sample of the same site before screening starts.
%
% Runs the model the policy was fitted for (its identity is checked by applySitePolicy), applies the site cut-off and reports
% sensitivity and specificity with Wilson 95% intervals and whether the screening targets (90% / 85%) are met as point estimates
% and as lower bounds. Counterpart of `calibrate_site.py verify`, with the same identity checks (row order does not matter):
% a policy that does not record its calibration images is refused, an image of the calibration sample is refused, and when the
% policy records patient identities, PatientIds (one per image) are required and shared patients are refused.
%
%   s = verifySitePolicy(paths, icdrGrades, "site_policy.json", "site_check.json", PatientIds=ids);
    arguments
        imagePaths (:,1) string
        grades (:,1) double {mustBeInteger, mustBeInRange(grades, 0, 4)}
        policyPath (1,1) string
        outputPath (1,1) string = ""
        options.PatientIds (:,1) string = strings(0, 1)
    end
    if numel(imagePaths) ~= numel(grades)
        error('NetrAI:SiteSampleMismatch', '%d images but %d grades.', numel(imagePaths), numel(grades));
    end
    if ~isempty(options.PatientIds) && numel(options.PatientIds) ~= numel(imagePaths)
        error('NetrAI:SiteSampleMismatch', '%d images but %d patient ids.', numel(imagePaths), numel(options.PatientIds));
    end
    baseDir = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(baseDir, 'src')));
    policy = jsondecode(fileread(policyPath));
    if isfield(policy, 'model_variant') && string(policy.model_variant) == "legacy"
        modelDir = fullfile(baseDir, 'models');
    else
        modelDir = fullfile(baseDir, 'variant_R3_E_768');
    end
    digests = strings(numel(imagePaths), 1);
    for i = 1:numel(imagePaths), digests(i) = string(fileSHA256(imagePaths(i))); end
    if ~isfield(policy, 'calibration_image_sha256') || isempty(policy.calibration_image_sha256)
        error('NetrAI:SitePolicyNoProvenance', ['The policy does not record its calibration images, so overlap with this sample ' ...
            'cannot be ruled out: re-create it with calibrateSite or python/calibrate_site.py score + fit.']);
    end
    shared = intersect(digests, string(policy.calibration_image_sha256));
    if ~isempty(shared)
        error('NetrAI:SiteSampleOverlap', '%d images were in the calibration sample: verification needs a separate sample.', numel(shared));
    end
    if isfield(policy, 'calibration_patient_sha256') && ~isempty(policy.calibration_patient_sha256)
        if isempty(options.PatientIds)
            error('NetrAI:SitePolicyNoProvenance', ['The calibration sample recorded patient identities: give PatientIds ' ...
                '(one per image) so that patients in both samples can be refused.']);
        end
        shared = intersect(textSHA256(options.PatientIds), string(policy.calibration_patient_sha256));
        if ~isempty(shared)
            error('NetrAI:SiteSampleOverlap', '%d patients appear in both the calibration and the verification sample.', numel(shared));
        end
    end
    [net, config] = loadNetrAIModel(string(fullfile(modelDir, 'netrai_fold0.onnx')), string(fullfile(modelDir, 'netrai_config.json')));
    [config, calibration] = applySitePolicy(config, policyPath);
    called = false(numel(imagePaths), 1);
    for i = 1:numel(imagePaths)
        x = preprocessModelInput(readScreeningImage(imagePaths(i)), config);
        [~, ~, called(i)] = predictDRGrade(net, x, config.grade_thresholds, config.referable_head_threshold);
    end
    summary = summarizeSiteVerification(called, grades >= 2);
    summary.site = calibration.site_name;
    summary.policy_sha256 = calibration.site_policy_sha256;
    if strlength(outputPath) > 0
        fid = fopen(outputPath, 'w'); cleanup = onCleanup(@() fclose(fid));
        fwrite(fid, jsonencode(summary, PrettyPrint=true));
    end
    fprintf('Verification: sensitivity %.3f (%.3f-%.3f), specificity %.3f (%.3f-%.3f); targets met: point %d, lower bound %d\n', ...
        summary.sensitivity, summary.sensitivity_ci95, summary.specificity, summary.specificity_ci95, ...
        summary.meets_targets_point, summary.meets_targets_lower_bound);
    if summary.referable < 40
        fprintf('WARNING: only %d referable cases: the sensitivity interval is wide (advised: at least 40).\n', summary.referable);
    end
end
