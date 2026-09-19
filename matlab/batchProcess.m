function summaryTable = batchProcess(folderPath, outputCsvPath, varargin)
%BATCHPROCESS Run NetrAI on an unlabeled image folder and export one CSV.
% Serial execution is intentional so this works in free MATLAB Online Basic,
% which does not include Parallel Computing Toolbox.

    baseDir = fileparts(mfilename('fullpath'));
    if nargin < 2 || strlength(string(outputCsvPath)) == 0
        outputCsvPath = fullfile(folderPath, 'batch_results.csv');
    end
    p = inputParser;
    addRequired(p, 'folderPath', @(x) ischar(x) || isstring(x));
    addRequired(p, 'outputCsvPath', @(x) ischar(x) || isstring(x));
    addParameter(p, 'ModelPath', fullfile(baseDir,'models','netrai_fold0.onnx'), @(x) ischar(x) || isstring(x));
    addParameter(p, 'ConfigPath', fullfile(baseDir,'models','netrai_config.json'), @(x) ischar(x) || isstring(x));
    addParameter(p, 'FullPipeline', true, @(x) islogical(x) && isscalar(x));
    parse(p, folderPath, outputCsvPath, varargin{:});
    folderPath = string(p.Results.folderPath);
    outputCsvPath = string(p.Results.outputCsvPath);
    if ~isfolder(folderPath), error('NetrAI:FolderNotFound', 'Folder not found: %s', folderPath); end
    outputParent = fileparts(outputCsvPath);
    if strlength(outputParent) > 0 && ~isfolder(outputParent), mkdir(outputParent); end

    extensions = ["*.png","*.jpg","*.jpeg","*.tif","*.tiff"];
    files = dir(fullfile(folderPath, extensions(1)));
    for i = 2:numel(extensions)
        files = [files; dir(fullfile(folderPath, extensions(i)))]; %#ok<AGROW>
    end
    [~, order] = sort(lower(string({files.name})));
    files = files(order); n = numel(files);
    if n == 0, summaryTable = table(); return; end

    filename = strings(n,1); status = repmat("pending",n,1); errorMessage = strings(n,1);
    iqs = nan(n,1); grade = nan(n,1); confidence = nan(n,1);
    isReferable = false(n,1); continuousScore = nan(n,1); processingSeconds = nan(n,1);
    probabilities = nan(n,5);
    referableHeadProbability = nan(n,1); headDisagreement = false(n,1);
    triageAction = repmat("",n,1);
    if ~p.Results.FullPipeline
        addpath(genpath(fullfile(baseDir,'src')));
        [net, config] = loadNetrAIModel(string(p.Results.ModelPath), string(p.Results.ConfigPath));
    end
    for i = 1:n
        filename(i) = string(files(i).name);
        imagePath = fullfile(string(files(i).folder), filename(i));
        timer = tic;
        try
            if p.Results.FullPipeline
                result = main_pipeline(imagePath, 'ModelPath',p.Results.ModelPath, ...
                    'ConfigPath',p.Results.ConfigPath, 'Verbose',false, 'GenerateReport',false, ...
                    'RunGradCAM',false, 'EnforceQuality',true);
                status(i) = result.status; iqs(i) = result.iqs;
                if result.status == "success"
                    grade(i) = result.grade; probabilities(i,:) = result.probabilities;
                    confidence(i) = result.confidence; isReferable(i) = result.isReferable;
                    continuousScore(i) = result.continuousScore;
                    referableHeadProbability(i) = result.referableHeadProbability;
                    headDisagreement(i) = result.headDisagreement;
                    triageAction(i) = result.triageAction;
                else
                    errorMessage(i) = string(result.qualityFeedback);
                    triageAction(i) = result.triageAction;
                end
            else
                img = imread(imagePath);
                if ndims(img)==2, img=repmat(img,1,1,3); end
                if size(img,3)==4, img=img(:,:,1:3); end
                [iqs(i),~] = assessImageQuality(im2uint8(img));
                input = preprocessModelInput(img,config);
                [grade(i),rawProbabilities,isReferable(i),continuousScore(i),~, ...
                    referableHeadProbability(i),referableHeadPositive,headDisagreement(i)] = ...
                    predictDRGrade(net,input,config.grade_thresholds, ...
                    config.referable_head_threshold);
                probabilities(i,:) = calibrateConfidence( ...
                    rawProbabilities, config.confidence_temperature);
                confidence(i) = probabilities(i,grade(i)+1); status(i) = "success";
                triageAction(i) = assignScreeningTriage( ...
                    false,isReferable(i),referableHeadPositive);
            end
        catch ME
            status(i) = "error";
            errorMessage(i) = string(ME.identifier) + ": " + string(ME.message);
        end
        processingSeconds(i) = toc(timer);
        fprintf('%d/%d %s: %s\n', i, n, filename(i), status(i));
    end
    summaryTable = table(filename,status,errorMessage,iqs,grade,confidence,isReferable, ...
        continuousScore,referableHeadProbability,headDisagreement,triageAction, ...
        probabilities(:,1),probabilities(:,2),probabilities(:,3), ...
        probabilities(:,4),probabilities(:,5),processingSeconds, ...
        'VariableNames', {'filename','status','error','iqs','grade','confidence','is_referable', ...
        'continuous_score','referable_head_probability','head_disagreement','triage_action', ...
        'prob_grade_0','prob_grade_1','prob_grade_2','prob_grade_3', ...
        'prob_grade_4','processing_seconds'});
    writetable(summaryTable, outputCsvPath);
    fprintf('Saved %d rows to %s (%d success, %d rejected, %d errors).\n', n, ...
        outputCsvPath, nnz(status=="success"), nnz(status=="rejected"), nnz(status=="error"));
end
