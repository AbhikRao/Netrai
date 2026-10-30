function policy = calibrateSite(imagePaths, grades, outputPath, options)
%CALIBRATESITE Fit and save a per-site referral cut-off in MATLAB (counterpart of python/calibrate_site.py).
%
% Runs the selected model on a labelled sample of ONE camera/site (ungradable images removed; ICDR grades 0-4; enrich the sample
% with referable cases: 40-80+ advised) and writes a site policy that main_pipeline(..., 'SitePolicy', outputPath) loads. The policy
% is bound to the model and config it was fitted for. Rule: highest cut-off with sample sensitivity >= 95% (confirmed on a fresh test).
% The policy records the SHA-256 of every calibration image and, with PatientIds, of every patient id (never the ids themselves),
% in the format python/calibrate_site.py shares, so that verification on the same images or patients is refused in both runtimes.
%
%   policy = calibrateSite(paths, grades, "site_policy.json", SiteName="PHC Nanded", Variant="E_768", PatientIds=ids);
%   Then check it on a separate sample: verifySitePolicy(otherPaths, otherGrades, "site_policy.json").
    arguments
        imagePaths (:,1) string
        grades (:,1) double {mustBeInteger, mustBeInRange(grades, 0, 4)}
        outputPath (1,1) string
        options.Variant (1,1) string {mustBeMember(options.Variant, ["E_768" "legacy"])} = "E_768"
        options.TargetSensitivity (1,1) double = 0.95
        options.SiteName (1,1) string = "unnamed site"
        options.Verbose (1,1) logical = true
        options.PatientIds (:,1) string = strings(0, 1)
    end
    if numel(imagePaths) ~= numel(grades)                           % before minutes of scoring, not after
        error('NetrAI:SiteSampleMismatch', '%d images but %d grades.', numel(imagePaths), numel(grades));
    end
    if ~isempty(options.PatientIds) && (numel(options.PatientIds) ~= numel(imagePaths) || any(strlength(strtrim(options.PatientIds)) == 0))
        error('NetrAI:SiteSampleMismatch', 'Give one non-empty patient id per image (%d images, %d ids).', numel(imagePaths), numel(options.PatientIds));
    end
    baseDir = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(baseDir, 'src')));
    if options.Variant == "E_768"
        modelDir = fullfile(baseDir, 'variant_R3_E_768');
    else
        modelDir = fullfile(baseDir, 'models');
    end
    modelPath = string(fullfile(modelDir, 'netrai_fold0.onnx'));
    configPath = string(fullfile(modelDir, 'netrai_config.json'));
    if ~isfile(modelPath)
        error('NetrAI:VariantMissing', 'Model weights not found: %s (E_768: run fetchE768Weights).', modelPath);
    end
    [net, config] = loadNetrAIModel(modelPath, configPath);
    scores = zeros(numel(imagePaths), 1);
    for i = 1:numel(imagePaths)
        rgb = readScreeningImage(imagePaths(i));
        modelInput = preprocessModelInput(rgb, config);
        [~, ~, ~, scores(i)] = predictDRGrade(net, modelInput, config.grade_thresholds, config.referable_head_threshold);
        if options.Verbose && mod(i, 25) == 0
            fprintf('calibrateSite: scored %d/%d images\n', i, numel(imagePaths));
        end
    end
    policy = fitSitePolicy(scores, grades, config, TargetSensitivity=options.TargetSensitivity, ...
        SiteName=options.SiteName, ModelVariant=options.Variant);
    digests = strings(numel(imagePaths), 1);
    for i = 1:numel(imagePaths), digests(i) = string(fileSHA256(imagePaths(i))); end
    policy.calibration_image_sha256 = cellstr(unique(digests));   % sorted; lets both verifiers refuse the calibration images
    if ~isempty(options.PatientIds)
        policy.calibration_patient_sha256 = cellstr(unique(textSHA256(options.PatientIds)));
    end
    fid = fopen(outputPath, 'w');
    if fid < 0
        error('NetrAI:SitePolicyWrite', 'Cannot write %s.', outputPath);
    end
    cleanup = onCleanup(@() fclose(fid));
    fwrite(fid, jsonencode(policy, PrettyPrint=true));
    if options.Verbose
        s = policy.site;
        fprintf('Site policy %s: boundary %.3f -> %.3f; sample sensitivity %.3f, specificity %.3f (%d referable / %d non-referable)\n', ...
            outputPath, s.base_boundary, s.fitted_boundary, s.sample_operating_point_fitted.sensitivity, ...
            s.sample_operating_point_fitted.specificity, s.referable, s.non_referable);
        for w = string(s.warnings(:)).', fprintf('WARNING: %s\n', w); end
    end
end
