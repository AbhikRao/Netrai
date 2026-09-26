function summary = validatePipeline(datasetPath, varargin)
%VALIDATEPIPELINE Batch validation with aggregate metrics and CSV/JSON export.
%
% summary = validatePipeline(datasetPath, Name=Value)
%   Mode="model" validates the ONNX classifier efficiently.
%   Mode="full" also runs quality and M2 lesion-candidate analysis.
%   SelectionCsv may contain an id_code column to select a fixed holdout set.

    baseDir = fileparts(mfilename('fullpath'));
    p = inputParser;
    addRequired(p, 'datasetPath', @(x) ischar(x) || isstring(x));
    addParameter(p, 'Mode', "model", @(x) any(strcmpi(string(x), ["model","full"])));
    addParameter(p, 'MaxImages', inf, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'StartIndex', 1, @(x) isnumeric(x) && isscalar(x) && x >= 1 && mod(x,1)==0);
    addParameter(p, 'SelectionCsv', "", @(x) ischar(x) || isstring(x));
    addParameter(p, 'OutputDir', fullfile(baseDir, 'results', 'validation'), @(x) ischar(x) || isstring(x));
    addParameter(p, 'ModelPath', fullfile(baseDir, 'models', 'netrai_fold0.onnx'), @(x) ischar(x) || isstring(x));
    addParameter(p, 'ConfigPath', fullfile(baseDir, 'models', 'netrai_config.json'), @(x) ischar(x) || isstring(x));
    parse(p, datasetPath, varargin{:});

    datasetPath = string(p.Results.datasetPath);
    mode = lower(string(p.Results.Mode));
    outputDir = string(p.Results.OutputDir);
    modelPath = string(p.Results.ModelPath);
    configPath = string(p.Results.ConfigPath);
    csvPath = fullfile(datasetPath, 'train.csv');
    imageDir = fullfile(datasetPath, 'train_images');
    if ~isfile(csvPath), error('NetrAI:LabelsNotFound', 'Missing %s', csvPath); end
    if ~isfolder(imageDir), error('NetrAI:ImagesNotFound', 'Missing %s', imageDir); end
    if ~isfolder(outputDir), mkdir(outputDir); end
    addpath(genpath(fullfile(baseDir, 'src')));

    labels = readtable(csvPath, 'TextType','string', 'VariableNamingRule','preserve');
    if ~all(ismember({'id_code','diagnosis'}, labels.Properties.VariableNames))
        error('NetrAI:InvalidLabels', 'train.csv must contain id_code and diagnosis.');
    end
    if any(ismissing(labels.id_code)) || numel(unique(labels.id_code)) ~= height(labels)
        error('NetrAI:InvalidIDs','Image IDs must be nonempty and unique.');
    end
    if any(~isfinite(labels.diagnosis) | labels.diagnosis ~= floor(labels.diagnosis) ...
            | labels.diagnosis < 0 | labels.diagnosis > 4)
        error('NetrAI:InvalidLabels','Labels must be finite integer grades 0-4.');
    end
    selectionCsv = string(p.Results.SelectionCsv);
    if strlength(selectionCsv) > 0
        selection = readtable(selectionCsv, 'TextType','string', 'VariableNamingRule','preserve');
        if ~ismember('id_code', selection.Properties.VariableNames)
            error('NetrAI:InvalidSelection', 'SelectionCsv must contain id_code.');
        end
        if numel(unique(selection.id_code)) ~= height(selection)
            error('NetrAI:InvalidSelection','Duplicate selection IDs.');
        end
        [found, positions] = ismember(selection.id_code, labels.id_code);
        if any(~found)
            error('NetrAI:InvalidSelection', '%d selected IDs are absent from train.csv.', nnz(~found));
        end
        labels = labels(positions, :);
    end
    startIndex = p.Results.StartIndex;
    if startIndex > height(labels)
        error('NetrAI:StartIndexOutOfRange', 'StartIndex %d exceeds %d selected rows.', startIndex, height(labels));
    end
    stopIndex = min(height(labels), startIndex + p.Results.MaxImages - 1);
    labels = labels(startIndex:stopIndex, :);
    n = height(labels);

    if mode == "model"
        [net, config] = loadNetrAIModel(modelPath, configPath);
    end
    idCode = labels.id_code;
    trueGrade = double(labels.diagnosis);
    predictedGrade = nan(n,1); continuousScore = nan(n,1);
    confidence = nan(n,1); referableProbability = nan(n,1); iqs = nan(n,1);
    probabilities = nan(n,5); rawProbabilities = nan(n,5);
    calibratedReferableProbability = nan(n,1); elapsedSeconds = nan(n,1);
    referableHeadProbability = nan(n,1); referableHeadPrediction = false(n,1);
    headDisagreement = false(n,1); triageAction = repmat("",n,1);
    status = repmat("pending", n, 1); errorMessage = repmat("", n, 1);
    maCount = nan(n,1); hemCount = nan(n,1); hardExudateCount = nan(n,1);
    softExudateCount = nan(n,1); nvDetected = nan(n,1); vesselDensity = nan(n,1);

    started = tic;
    for i = 1:n
        imagePath = resolveImage(imageDir, idCode(i));
        if strlength(imagePath) == 0
            status(i) = "error"; errorMessage(i) = "image file not found";
            continue;
        end
        itemTimer = tic;
        try
            img = imread(imagePath);
            if ndims(img) == 2, img = repmat(img,1,1,3); end
            if size(img,3) == 4, img = img(:,:,1:3); end
            [iqs(i), ~] = assessImageQuality(im2uint8(img));
            if mode == "model"
                input = preprocessModelInput(img, config);
                [predictedGrade(i), rawProbabilities(i,:), gradeReferable, ...
                    continuousScore(i), ~, referableHeadProbability(i), ...
                    referableHeadPrediction(i), headDisagreement(i)] = ...
                    predictDRGrade(net, input, config.grade_thresholds, ...
                    config.referable_head_threshold);
                probabilities(i,:) = calibrateConfidence( ...
                    rawProbabilities(i,:), config.confidence_temperature);
                triageAction(i) = assignScreeningTriage( ...
                    false, gradeReferable, referableHeadPrediction(i));
            else
                result = main_pipeline(imagePath, 'ModelPath',modelPath, 'ConfigPath',configPath, ...
                    'Verbose',false, 'GenerateReport',false, 'RunGradCAM',false, 'EnforceQuality',false);
                predictedGrade(i) = result.grade;
                rawProbabilities(i,:) = result.rawProbabilities;
                probabilities(i,:) = result.probabilities;
                continuousScore(i) = result.continuousScore;
                referableHeadProbability(i) = result.referableHeadProbability;
                referableHeadPrediction(i) = result.referableHeadPositive;
                headDisagreement(i) = result.headDisagreement;
                triageAction(i) = result.triageAction;
                maCount(i) = result.segResults.maCount;
                hemCount(i) = result.segResults.hemCount;
                hardExudateCount(i) = result.segResults.hardCount;
                softExudateCount(i) = result.segResults.softCount;
                nvDetected(i) = double(result.segResults.nvDetected);
                vesselDensity(i) = nnz(result.segResults.vesselMask) / ...
                    max(nnz(result.segResults.fovMask), 1);
            end
            confidence(i) = probabilities(i, predictedGrade(i)+1);
            referableProbability(i) = sum(rawProbabilities(i,3:5));
            calibratedReferableProbability(i) = sum(probabilities(i,3:5));
            status(i) = "success";
        catch ME
            status(i) = "error";
            errorMessage(i) = string(ME.identifier) + ": " + string(ME.message);
        end
        elapsedSeconds(i) = toc(itemTimer);
        if mod(i,25) == 0 || i == n
            fprintf('Validated %d/%d (%d errors)\n', i, n, nnz(status=="error"));
        end
    end

    perImage = table(idCode, trueGrade, predictedGrade, continuousScore, confidence, ...
        referableProbability, calibratedReferableProbability, ...
        referableHeadProbability, referableHeadPrediction, ...
        headDisagreement, triageAction, ...
        probabilities(:,1), probabilities(:,2), probabilities(:,3), ...
        probabilities(:,4), probabilities(:,5), rawProbabilities(:,1), ...
        rawProbabilities(:,2), rawProbabilities(:,3), rawProbabilities(:,4), ...
        rawProbabilities(:,5), iqs, elapsedSeconds, status, errorMessage, ...
        maCount, hemCount, hardExudateCount, softExudateCount, nvDetected, vesselDensity, ...
        'VariableNames', {'id_code','true_grade','predicted_grade','continuous_score', ...
        'confidence','referable_probability','calibrated_referable_probability', ...
        'referable_head_probability','referable_head_prediction', ...
        'head_disagreement','triage_action', ...
        'prob_grade_0','prob_grade_1','prob_grade_2','prob_grade_3','prob_grade_4', ...
        'raw_prob_grade_0','raw_prob_grade_1','raw_prob_grade_2','raw_prob_grade_3', ...
        'raw_prob_grade_4','iqs','elapsed_seconds','status','error', ...
        'ma_count','hem_count','hard_exudate_count','soft_exudate_count','nv_detected','vessel_density'});
    writetable(perImage, fullfile(outputDir, 'per_image_results.csv'));
    provenance = struct('schema_version',1,'model_sha256',fileSHA256(modelPath), ...
        'config_sha256',fileSHA256(configPath),'labels_sha256',fileSHA256(csvPath), ...
        'mode',mode,'selection_sha256',"",'preprocess_sha256', ...
        fileSHA256(fullfile(baseDir,'src','m3_grading','preprocessModelInput.m')));
    if strlength(selectionCsv)>0, provenance.selection_sha256=fileSHA256(selectionCsv); end
    fid=fopen(fullfile(outputDir,'run_identity.json'),'w');
    if fid<0, error('NetrAI:WriteFailed','Cannot write run identity'); end
    fprintf(fid,'%s\n',jsonencode(provenance)); fclose(fid);

    valid = status == "success";
    if ~any(valid), error('NetrAI:NoValidResults', 'No images completed successfully.'); end
    yTrue = trueGrade(valid); yPred = predictedGrade(valid);
    confMat = zeros(5,5);
    for i = 1:numel(yTrue)
        confMat(yTrue(i)+1, yPred(i)+1) = confMat(yTrue(i)+1, yPred(i)+1) + 1;
    end
    writematrix(confMat, fullfile(outputDir, 'confusion_matrix.csv'));

    classMetrics = computeMetrics(confMat);
    qwk = quadraticKappa(confMat);
    refTrue = yTrue >= 2; refPred = yPred >= 2;
    tp = nnz(refTrue & refPred); tn = nnz(~refTrue & ~refPred);
    fp = nnz(~refTrue & refPred); fn = nnz(refTrue & ~refPred);
    sensitivity = safeDivide(tp, tp+fn); specificity = safeDivide(tn, tn+fp);
    precision = safeDivide(tp, tp+fp); refAccuracy = (tp+tn)/numel(yTrue);
    refF1 = safeDivide(2*precision*sensitivity, precision+sensitivity);
    auc = binaryAUC(refTrue, referableProbability(valid));
    independentPred = referableHeadPrediction(valid);
    independentScores = referableHeadProbability(valid);
    itp = nnz(refTrue & independentPred); itn = nnz(~refTrue & ~independentPred);
    ifp = nnz(~refTrue & independentPred); ifn = nnz(refTrue & ~independentPred);
    independentSensitivity = safeDivide(itp, itp+ifn);
    independentSpecificity = safeDivide(itn, itn+ifp);
    independentAuc = binaryAUC(refTrue, independentScores);
    disagreements = headDisagreement(valid);
    concordant = ~disagreements;
    concordantSensitivity = safeDivide( ...
        nnz(refTrue(concordant) & refPred(concordant)), nnz(refTrue(concordant)));
    concordantSpecificity = safeDivide( ...
        nnz(~refTrue(concordant) & ~refPred(concordant)), nnz(~refTrue(concordant)));

    summary = struct( ...
        'requested', n, 'successful', nnz(valid), 'errors', nnz(~valid), ...
        'coverage', nnz(valid)/n, 'mode', mode, 'start_index', startIndex, ...
        'stop_index', stopIndex, 'qwk', qwk, ...
        'accuracy', mean(yTrue==yPred), 'mean_absolute_error', mean(abs(yTrue-yPred)), ...
        'within_one_grade_accuracy', mean(abs(yTrue-yPred)<=1), ...
        'macro_f1', classMetrics.macroF1, 'referable_sensitivity', sensitivity, ...
        'referable_specificity', specificity, 'referable_precision', precision, ...
        'referable_accuracy', refAccuracy, 'referable_f1', refF1, ...
        'referable_auc', auc, ...
        'referable_head_sensitivity', independentSensitivity, ...
        'referable_head_specificity', independentSpecificity, ...
        'referable_head_auc', independentAuc, ...
        'head_disagreement_count', nnz(disagreements), ...
        'head_disagreement_rate', mean(disagreements), ...
        'concordant_coverage', mean(concordant), ...
        'concordant_sensitivity', concordantSensitivity, ...
        'concordant_specificity', concordantSpecificity, ...
        'total_seconds', toc(started), ...
        'average_seconds', mean(elapsedSeconds(valid)));

    metricNames = string(fieldnames(summary));
    metricValues = strings(numel(metricNames),1);
    for i = 1:numel(metricNames)
        value = summary.(metricNames(i));
        if isnumeric(value) || islogical(value)
            metricValues(i) = string(value);
        else
            metricValues(i) = string(value);
        end
    end
    writetable(table(metricNames, metricValues, 'VariableNames',{'metric','value'}), ...
        fullfile(outputDir, 'metrics_summary.csv'));
    fid = fopen(fullfile(outputDir, 'aggregate_metrics.json'), 'w');
    if fid < 0, error('NetrAI:WriteFailed', 'Cannot create aggregate_metrics.json'); end
    cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%s\n', jsonencode(summary, 'PrettyPrint',true));

    grade = (0:4)'; support = sum(confMat,2);
    sensitivityGrade = classMetrics.sensitivity'; specificityGrade = classMetrics.specificity';
    precisionGrade = classMetrics.precision'; f1Grade = classMetrics.f1Score';
    writetable(table(grade,support,sensitivityGrade,specificityGrade,precisionGrade,f1Grade), ...
        fullfile(outputDir, 'per_grade_metrics.csv'));
    save(fullfile(outputDir, 'validation_results.mat'), 'summary', 'perImage', 'confMat');
    fprintf('QWK %.4f | sensitivity %.2f%% | specificity %.2f%% | AUC %.4f | errors %d\n', ...
        qwk, 100*sensitivity, 100*specificity, auc, nnz(~valid));
end


function path = resolveImage(imageDir, imageId)
    path = "";
    extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff"];
    for ext = extensions
        candidate = fullfile(imageDir, imageId + ext);
        if isfile(candidate), path = string(candidate); return; end
    end
end


function value = safeDivide(numerator, denominator)
    if denominator == 0, value = NaN; else, value = numerator/denominator; end
end


function kappa = quadraticKappa(confMat)
    n = size(confMat,1); total = sum(confMat,'all');
    expected = sum(confMat,2) * sum(confMat,1) / total;
    [row,col] = ndgrid(0:n-1,0:n-1);
    weights = ((row-col).^2)/(n-1)^2;
    kappa = 1 - sum(weights.*confMat,'all')/sum(weights.*expected,'all');
end


function auc = binaryAUC(labels, scores)
    labels = logical(labels(:)); scores = scores(:);
    positives = nnz(labels); negatives = nnz(~labels);
    if positives == 0 || negatives == 0, auc = NaN; return; end
    [sortedScores, order] = sort(scores, 'ascend');
    ranks = (1:numel(scores))';
    first = 1;
    while first <= numel(scores)
        last = first;
        while last < numel(scores) && sortedScores(last+1) == sortedScores(first)
            last = last + 1;
        end
        ranks(first:last) = mean(first:last);
        first = last + 1;
    end
    originalRanks = zeros(size(ranks)); originalRanks(order) = ranks;
    auc = (sum(originalRanks(labels)) - positives*(positives+1)/2) / (positives*negatives);
end
