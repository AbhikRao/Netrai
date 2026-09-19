function report = verifyMatlabPackage()
%VERIFYMATLABPACKAGE Verify dependencies, ONNX import, and reference parity.
% Run this once after uploading the project to MATLAB Online.

    baseDir = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(baseDir, 'src')));
    modelPath = string(fullfile(baseDir, 'models', 'netrai_fold0.onnx'));
    configPath = string(fullfile(baseDir, 'models', 'netrai_config.json'));
    fixturePath = fullfile(baseDir, 'models', 'parity_fixture.mat');
    preprocessingReferencePath = fullfile(baseDir, 'models', 'preprocessing_reference.json');

    products = ver;
    installed = string({products.Name});
    required = ["MATLAB", "Deep Learning Toolbox", "Image Processing Toolbox"];
    missing = required(~ismember(required, installed));
    if ~isempty(missing)
        error('NetrAI:MissingProducts', 'Missing required products: %s', strjoin(missing, ', '));
    end
    if isempty(which('importNetworkFromONNX'))
        error('NetrAI:MissingONNXConverter', [ ...
            'Install "Deep Learning Toolbox Converter for ONNX Model ' ...
            'Format" from Home > Add-Ons, then rerun this verifier.']);
    end
    if ~isfile(fixturePath)
        error('NetrAI:FixtureMissing', 'Missing parity fixture: %s', fixturePath);
    end

    [net, config] = loadNetrAIModel(modelPath, configPath);
    fixture = load(fixturePath);
    [grade, probabilities, isReferable, score, logits, referableProbability, ...
        referableHeadPositive, headDisagreement, referableLogit] = ...
        predictDRGrade(net, fixture.model_input, config.grade_thresholds, ...
        config.referable_head_threshold);
    maxLogitError = max(abs(logits - double(fixture.expected_logits(:)')));
    tolerance = 0.02;
    if maxLogitError > tolerance
        error('NetrAI:ParityFailed', ...
            'MATLAB/ONNX logit difference %.6f exceeds %.6f.', maxLogitError, tolerance);
    end
    referableLogitError = abs(referableLogit - ...
        double(fixture.expected_referable_logit));
    if referableLogitError > tolerance
        error('NetrAI:ReferableParityFailed', ...
            'MATLAB/ONNX referable-logit difference %.6f exceeds %.6f.', ...
            referableLogitError, tolerance);
    end
    if grade ~= double(fixture.expected_grade)
        error('NetrAI:GradeParityFailed', 'Expected grade %d but MATLAB returned %d.', ...
            fixture.expected_grade, grade);
    end
    if abs(sum(probabilities)-1) > 1e-6 || any(probabilities < 0)
        error('NetrAI:ProbabilityInvariant', 'Model probabilities are invalid.');
    end
    calibrated = calibrateConfidence(probabilities, config.confidence_temperature);
    if abs(sum(calibrated)-1) > 1e-6 || any(calibrated < 0)
        error('NetrAI:CalibrationInvariant', 'Calibrated probabilities are invalid.');
    end

    preprocessingLogitError = NaN;
    endToEndGradePassed = false;
    if isfile(preprocessingReferencePath)
        reference = jsondecode(fileread(preprocessingReferencePath));
        smokePath = fullfile(fileparts(baseDir), reference.image_relative_to_project);
        if isfile(smokePath)
            smokeImage = imread(smokePath);
            smokeInput = preprocessModelInput(smokeImage, config);
            [smokeGrade, ~, ~, ~, smokeLogits, smokeReferableProbability] = ...
                predictDRGrade(net, smokeInput, config.grade_thresholds, ...
                config.referable_head_threshold);
            preprocessingLogitError = max(abs(smokeLogits-double(reference.expected_logits(:)')));
            endToEndGradePassed = smokeGrade == double(reference.expected_grade);
            if ~endToEndGradePassed
                error('NetrAI:PreprocessingParityFailed', ...
                    'End-to-end expected grade %d but MATLAB returned %d.', ...
                    reference.expected_grade, smokeGrade);
            end
            if preprocessingLogitError > 0.75
                warning('NetrAI:PreprocessingDrift', [ ...
                    'End-to-end grade matches, but logit difference %.4f is large. ' ...
                    'Review MATLAB/OpenCV resize and Gaussian boundary behavior.'], ...
                    preprocessingLogitError);
            end
            if abs(smokeReferableProbability - ...
                    double(reference.expected_referable_head_probability)) > 0.10
                warning('NetrAI:ReferablePreprocessingDrift', ...
                    'End-to-end referable-head probability differs by more than 0.10.');
            end
        end
    end

    report = struct('passed', true, 'matlab_release', version('-release'), ...
        'max_logit_error', maxLogitError, 'tolerance', tolerance, ...
        'referable_logit_error', referableLogitError, ...
        'grade', grade, 'continuous_score', score, 'is_referable', isReferable, ...
        'referable_head_probability', referableProbability, ...
        'referable_head_positive', referableHeadPositive, ...
        'head_disagreement', headDisagreement, ...
        'confidence_temperature', config.confidence_temperature, ...
        'end_to_end_grade_passed', endToEndGradePassed, ...
        'preprocessing_max_logit_error', preprocessingLogitError);
    fprintf('NetrAI MATLAB package PASS\n');
    fprintf('Release %s | max ONNX logit error %.8f | grade %d | score %.4f\n', ...
        report.matlab_release, maxLogitError, grade, score);
    if endToEndGradePassed
        fprintf('End-to-end preprocessing grade PASS | max logit difference %.6f\n', ...
            preprocessingLogitError);
    end
end
