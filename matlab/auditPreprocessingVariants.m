function [perImage, aggregate] = auditPreprocessingVariants(fixtureDir, varargin)
%AUDITPREPROCESSINGVARIANTS Compare MATLAB preprocessing variants with Python.
% Each MAT fixture must contain image_path, expected_input, expected_score,
% expected_raw_probabilities, expected_referable_head_probability, and
% expected_grade. This diagnostic never changes the production pipeline.

    baseDir = fileparts(mfilename('fullpath'));
    p = inputParser;
    addRequired(p, 'fixtureDir', @(x) ischar(x) || isstring(x));
    addParameter(p, 'OutputDir', fullfile(baseDir, 'results', 'preprocessing_audit'), ...
        @(x) ischar(x) || isstring(x));
    addParameter(p, 'ModelPath', fullfile(baseDir, 'models', 'netrai_fold0.onnx'), ...
        @(x) ischar(x) || isstring(x));
    addParameter(p, 'ConfigPath', fullfile(baseDir, 'models', 'netrai_config.json'), ...
        @(x) ischar(x) || isstring(x));
    parse(p, fixtureDir, varargin{:});

    fixtureDir = string(p.Results.fixtureDir);
    outputDir = string(p.Results.OutputDir);
    if ~isfolder(fixtureDir)
        error('NetrAI:FixtureDirectoryMissing', 'Missing fixture directory: %s', fixtureDir);
    end
    if ~isfolder(outputDir), mkdir(outputDir); end
    addpath(genpath(fullfile(baseDir, 'src')));
    [net, config] = loadNetrAIModel(string(p.Results.ModelPath), string(p.Results.ConfigPath));

    files = dir(fullfile(fixtureDir, '*.mat'));
    if isempty(files)
        error('NetrAI:NoFixtures', 'No MAT fixtures found in %s', fixtureDir);
    end
    methods = ["box", "bilinear", "bicubic"];
    paddings = ["replicate", "symmetric"];
    quantized = [false, true];
    row = 0;
    count = numel(files) * numel(methods) * numel(paddings) * numel(quantized);
    idCode = strings(count,1); variant = strings(count,1);
    inputMeanAbsDifference = nan(count,1); inputMaxAbsDifference = nan(count,1);
    scoreAbsDifference = nan(count,1); maxProbabilityAbsDifference = nan(count,1);
    referableHeadAbsDifference = nan(count,1); gradeMatch = false(count,1);

    for fileIndex = 1:numel(files)
        fixture = load(fullfile(files(fileIndex).folder, files(fileIndex).name));
        required = {'id_code','image_path','expected_input','expected_score', ...
            'expected_raw_probabilities','expected_referable_head_probability', ...
            'expected_grade'};
        for fieldIndex = 1:numel(required)
            if ~isfield(fixture, required{fieldIndex})
                error('NetrAI:InvalidFixture', '%s is missing %s.', ...
                    files(fileIndex).name, required{fieldIndex});
            end
        end
        image = imread(string(fixture.image_path));
        expectedInput = single(fixture.expected_input);
        for method = methods
            for padding = paddings
                for useQuantizedStages = quantized
                    row = row + 1;
                    candidate = preprocessVariant( ...
                        image, config, method, padding, useQuantizedStages);
                    [grade, probabilities, ~, score, ~, referableProbability] = ...
                        predictDRGrade(net, candidate, config.grade_thresholds, ...
                        config.referable_head_threshold);
                    difference = abs(candidate - expectedInput);
                    idCode(row) = string(fixture.id_code);
                    variant(row) = method + "_" + padding + "_" + ...
                        string(conditional(useQuantizedStages, "uint8", "float"));
                    inputMeanAbsDifference(row) = mean(difference, 'all');
                    inputMaxAbsDifference(row) = max(difference, [], 'all');
                    scoreAbsDifference(row) = abs(score - double(fixture.expected_score));
                    maxProbabilityAbsDifference(row) = max(abs( ...
                        probabilities - double(fixture.expected_raw_probabilities(:)')));
                    referableHeadAbsDifference(row) = abs(referableProbability - ...
                        double(fixture.expected_referable_head_probability));
                    gradeMatch(row) = grade == double(fixture.expected_grade);
                end
            end
        end
    end

    perImage = table(idCode, variant, inputMeanAbsDifference, ...
        inputMaxAbsDifference, scoreAbsDifference, maxProbabilityAbsDifference, ...
        referableHeadAbsDifference, gradeMatch);
    writetable(perImage, fullfile(outputDir, 'preprocessing_variant_per_image.csv'));
    aggregate = groupsummary(perImage, 'variant', {'mean','max'}, ...
        {'inputMeanAbsDifference','inputMaxAbsDifference','scoreAbsDifference', ...
        'maxProbabilityAbsDifference','referableHeadAbsDifference'});
    gradeRates = groupsummary(perImage, 'variant', 'mean', 'gradeMatch');
    aggregate.grade_match_rate = gradeRates.mean_gradeMatch;
    aggregate = sortrows(aggregate, 'mean_scoreAbsDifference', 'ascend');
    writetable(aggregate, fullfile(outputDir, 'preprocessing_variant_summary.csv'));
    save(fullfile(outputDir, 'preprocessing_variant_audit.mat'), 'perImage', 'aggregate');
    disp(aggregate);
end


function output = preprocessVariant(imageRGB, config, method, padding, quantizeStages)
    if ismatrix(imageRGB), imageRGB = repmat(imageRGB, 1, 1, 3); end
    if size(imageRGB,3) == 4, imageRGB = imageRGB(:,:,1:3); end
    imageRGB = im2uint8(imageRGB);
    foreground = max(imageRGB, [], 3) > double(config.black_crop_threshold);
    if any(foreground(:))
        rows = find(any(foreground, 2)); cols = find(any(foreground, 1));
        imageRGB = imageRGB(rows(1):rows(end), cols(1):cols(end), :);
    end
    target = double(config.input_size);
    resized = imresize(imageRGB, [target target], char(method));
    sigma = double(config.ben_graham_sigma);
    filterSize = 2*ceil(3*sigma)+1;
    if quantizeStages
        blurred = imgaussfilt(resized, sigma, 'FilterSize',filterSize, ...
            'Padding',char(padding));
        enhanced = imlincomb(4, resized, -4, blurred, 128, 'uint8');
        enhanced = single(enhanced) ./ 255;
    else
        blurred = imgaussfilt(single(resized), sigma, 'FilterSize',filterSize, ...
            'Padding',char(padding));
        enhanced = 4 .* single(resized) - 4 .* blurred + 128;
        enhanced = min(max(enhanced, 0), 255) ./ 255;
    end
    means = reshape(single(config.imagenet_mean), 1, 1, 3);
    stds = reshape(single(config.imagenet_std), 1, 1, 3);
    output = (enhanced - means) ./ stds;
end


function value = conditional(condition, trueValue, falseValue)
    if condition, value = trueValue; else, value = falseValue; end
end
