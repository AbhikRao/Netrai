function results = main_pipeline(imagePath, varargin)
%MAIN_PIPELINE Run the native MATLAB NetrAI screening pipeline.
%
% The trained EfficientNet-B4 grade head is loaded from ONNX. Anatomical and
% lesion analysis remain native MATLAB/Image Processing Toolbox operations.
% No model, segmentation, report, or explanation failure is replaced with a
% fabricated clinical result.

    baseDir = fileparts(mfilename('fullpath'));
    defaultModel = fullfile(baseDir, 'models', 'netrai_fold0.onnx');
    defaultConfig = fullfile(baseDir, 'models', 'netrai_config.json');

    p = inputParser;
    addRequired(p, 'imagePath', @(x) ischar(x) || isstring(x));
    addParameter(p, 'ModelPath', defaultModel, @(x) ischar(x) || isstring(x));
    addParameter(p, 'ConfigPath', defaultConfig, @(x) ischar(x) || isstring(x));
    addParameter(p, 'OutputDir', fullfile(baseDir, 'results'), @(x) ischar(x) || isstring(x));
    addParameter(p, 'Verbose', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'GenerateReport', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'RunGradCAM', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'EnforceQuality', true, @(x) islogical(x) && isscalar(x));
    parse(p, imagePath, varargin{:});

    imagePath = string(p.Results.imagePath);
    modelPath = string(p.Results.ModelPath);
    configPath = string(p.Results.ConfigPath);
    outDir = string(p.Results.OutputDir);
    verbose = p.Results.Verbose;
    generateReport = p.Results.GenerateReport;
    runGradCAM = p.Results.RunGradCAM;
    enforceQuality = p.Results.EnforceQuality;

    addpath(genpath(fullfile(baseDir, 'src')));
    if ~isfile(imagePath)
        error('NetrAI:ImageNotFound', 'Input image not found: %s', imagePath);
    end
    if generateReport && ~isfolder(outDir)
        mkdir(outDir);
    end

    rawImg = imread(imagePath);
    if ndims(rawImg) == 2
        rawImg = repmat(rawImg, 1, 1, 3);
    elseif size(rawImg, 3) == 4
        rawImg = rawImg(:, :, 1:3);
    end
    rawImg = im2uint8(rawImg);

    [iqs, qualityMetrics] = assessImageQuality(rawImg);
    qualityFeedback = generateRecaptureFeedback(qualityMetrics);
    if enforceQuality && iqs < 0.3
        results = struct( ...
            'status', "rejected", 'imagePath', imagePath, 'iqs', iqs, ...
            'qualityMetrics', qualityMetrics, 'qualityFeedback', qualityFeedback, ...
            'grade', NaN, 'probabilities', nan(1, 5), 'rawProbabilities', nan(1, 5), ...
            'confidence', NaN, 'isReferable', false, 'continuousScore', NaN, ...
            'referableHeadProbability', NaN, 'referableHeadPositive', false, ...
            'headDisagreement', false, 'triageAction', "recapture", ...
            'segResults', struct(), 'gradcamStatus', "not run", 'gradcamQC', struct(), ...
            'reportPath', "");
        if verbose
            fprintf('Rejected: IQS %.3f. %s\n', iqs, qualityFeedback);
        end
        return;
    end

    % Enhancement is only for lesion analysis. The classifier always sees the
    % original image because that is the distribution used during training.
    analysisImg = rawImg;
    if iqs < 0.7
        analysisImg = enhanceFundusImage(rawImg);
    end

    [procImg, greenCh, fovMask] = preprocessFundusImage(analysisImg);
    [odCenter, odRadius] = localizeOpticDisc(procImg, greenCh, fovMask);
    foveaCenter = localizeFovea(greenCh, odCenter, odRadius);
    vesselMask = segmentVessels(greenCh, fovMask);
    [maMask, maCount, maCentroids] = detectMicroaneurysms( ...
        greenCh, vesselMask, fovMask, odCenter, odRadius);
    [hemMask, hemCount, hemStats] = detectHemorrhages( ...
        greenCh, vesselMask, maMask, fovMask, odCenter, odRadius);
    [exMask, hardCount, softCount, foveaDist] = detectExudates( ...
        procImg, greenCh, foveaCenter, odCenter, odRadius, fovMask);
    [nvMask, nvDetected] = detectNeovascularization( ...
        greenCh, vesselMask, odCenter, odRadius, fovMask);

    segResults = struct( ...
        'fovMask', fovMask, 'odCenter', odCenter, 'odRadius', odRadius, ...
        'foveaCenter', foveaCenter, 'vesselMask', vesselMask, ...
        'maMask', maMask, 'maCount', maCount, 'maCentroids', maCentroids, ...
        'hemMask', hemMask, 'hemCount', hemCount, 'hemStats', hemStats, ...
        'exMask', exMask, 'hardCount', hardCount, 'softCount', softCount, ...
        'foveaDist', foveaDist, 'nvMask', nvMask, 'nvDetected', nvDetected);
    clinicalFeatures = extractClinicalFeatures(segResults);

    [net, config] = loadNetrAIModel(modelPath, configPath);
    [modelInput, modelFov] = preprocessModelInput(rawImg, config);
    modelDisplay = min(max(double(modelInput) .* reshape([0.229 0.224 0.225],1,1,3) ...
        + reshape([0.485 0.456 0.406],1,1,3),0),1);
    [grade, rawProbs, isReferable, continuousScore, logits, ...
        referableHeadProbability, referableHeadPositive, headDisagreement, ...
        referableLogit] = predictDRGrade(net, modelInput, ...
        config.grade_thresholds, config.referable_head_threshold);
    [calProbs, temperature] = calibrateConfidence( ...
        rawProbs, config.confidence_temperature);
    confidence = calProbs(grade + 1);
    triageAction = assignScreeningTriage( ...
        false, isReferable, referableHeadPositive);

    if runGradCAM
        [heatmap, gradcamStatus] = generateGradCAM(net, modelInput, 6);
    else
        heatmap = [];
        gradcamStatus = "disabled";
    end
    gradcamQC = struct();
    if ~isempty(heatmap)
        gradcamQC = assessGradCAMQuality(heatmap, modelFov);
        if gradcamQC.shortcut_flag
            gradcamStatus = "available - QC FLAG: review corner/FOV attention";
        end
    end
    checklist = generateEvidenceChecklist(segResults, grade, calProbs);
    overlayImg = generateLesionOverlay(procImg, segResults);

    reportPath = "";
    if generateReport
        safety = struct('referableHeadProbability', referableHeadProbability, ...
            'referableHeadPositive', referableHeadPositive, ...
            'headDisagreement', headDisagreement, 'triageAction', triageAction);
        reportPath = string(generateClinicalReport(procImg, segResults, grade, ...
            calProbs, checklist, heatmap, overlayImg, outDir, gradcamStatus, ...
            gradcamQC, safety, modelDisplay));
    end

    if verbose
        showResults(procImg, overlayImg, heatmap, gradcamStatus, calProbs, grade, modelDisplay);
        fprintf('\n--- NetrAI result ---\n');
        fprintf('Grade: %d | continuous score: %.4f | confidence: %.2f%%\n', ...
            grade, continuousScore, confidence * 100);
        fprintf('Referable DR: %s | IQS: %.3f\n', string(isReferable), iqs);
        fprintf('Referable head: %.2f%% | disagreement: %s | triage: %s\n', ...
            referableHeadProbability * 100, string(headDisagreement), triageAction);
        fprintf('Grad-CAM: %s\n', gradcamStatus);
    end

    results = struct( ...
        'status', "success", 'imagePath', imagePath, 'iqs', iqs, ...
        'qualityMetrics', qualityMetrics, 'qualityFeedback', qualityFeedback, ...
        'grade', grade, 'probabilities', calProbs, 'rawProbabilities', rawProbs, ...
        'confidence', confidence, 'isReferable', isReferable, ...
        'continuousScore', continuousScore, 'logits', logits, ...
        'referableLogit', referableLogit, ...
        'gradeHeadReferableProbability', sum(rawProbs(3:5)), ...
        'referableHeadProbability', referableHeadProbability, ...
        'referableHeadPositive', referableHeadPositive, ...
        'headDisagreement', headDisagreement, 'triageAction', triageAction, ...
        'clinicalFeatures', clinicalFeatures, 'temperature', temperature, ...
        'segResults', segResults, 'gradcamStatus', gradcamStatus, ...
        'gradcamQC', gradcamQC, 'explanationTarget',"referral_logit", ...
        'researchOnly',true, ...
        'reportPath', reportPath);
end


function showResults(procImg, overlayImg, heatmap, gradcamStatus, probabilities, grade, modelDisplay)
    figure('Name', 'NetrAI Analysis Results', 'Position', [100, 100, 1100, 800]);
    tiledlayout(2, 2, 'TileSpacing', 'compact');
    nexttile; imshow(procImg); title('Analysis Image');
    nexttile; imshow(overlayImg); title('Candidate Lesion Overlay');
    nexttile;
    if isempty(heatmap)
        imshow(procImg); title("Grad-CAM " + gradcamStatus, 'Interpreter', 'none');
    else
        imshow(modelDisplay); hold on;
        h = imagesc(imresize(heatmap, size(modelDisplay, [1 2])));
        colormap(gca, jet); alpha(h, 0.4); axis image off;
        title('Genuine Grad-CAM Attention');
    end
    nexttile;
    b = bar(0:4, probabilities); b.FaceColor = 'flat';
    b.CData(grade + 1, :) = [0.85 0.32 0.09];
    xticks(0:4); xticklabels({'No DR','Mild','Moderate','Severe','PDR'});
    ylabel('Calibrated confidence'); ylim([0 1]); title('DR Grade Probabilities');
end
