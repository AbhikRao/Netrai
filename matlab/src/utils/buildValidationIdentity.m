function [identity, imagePaths] = buildValidationIdentity(baseDir, modelPath, configPath, labelsPath, selectionPath, mode, cohort)
%BUILDVALIDATIONIDENTITY Snapshot artifacts and complete cohort before inference.
    [bundle,modelDigest] = verifyDeploymentBundle(string(modelPath),string(configPath));
    configDigest=bundle.artifacts(strcmp({bundle.artifacts.role},'config')).sha256;
    stream=fopen(configPath,'rb');
    if stream<0, error('NetrAI:MissingArtifact','Cannot read deployment configuration.'); end
    closeStream=onCleanup(@() fclose(stream));
    configBytes=fread(stream,inf,'*uint8'); clear closeStream;
    if ~strcmp(sha256Bytes(configBytes),configDigest)
        error('NetrAI:ChangedArtifact','Configuration changed while capturing validation policy.');
    end
    configJson=native2unicode(configBytes(:)','UTF-8');
    if ~strcmp(sha256Bytes(unicode2native(configJson,'UTF-8')),configDigest)
        error('NetrAI:InvalidConfig','Deployment configuration must use valid UTF-8.');
    end
    config=jsondecode(configJson);
    policy=decisionPolicy(config);
    sources = dir(fullfile(baseDir,'src','**','*.m'));
    sources = [sources; dir(fullfile(baseDir,'main_pipeline.m')); ...
        dir(fullfile(baseDir,'validatePipeline.m')); dir(fullfile(baseDir,'aggregateValidationRuns.m'))];
    sourceFiles = repmat(struct('path','','sha256',''),numel(sources),1);
    for i = 1:numel(sources)
        path = fullfile(sources(i).folder,sources(i).name);
        relative = extractAfter(string(path),strlength(string(baseDir))+1);
        sourceFiles(i) = struct('path',char(relative),'sha256',fileSHA256(path));
    end
    [~,order] = sort(string({sourceFiles.path})); sourceFiles = sourceFiles(order);
    imagePaths = strings(height(cohort),1);
    expected = repmat(struct('id_code','','true_grade',0,'image_sha256',''),height(cohort),1);
    imageDir = fullfile(fileparts(labelsPath),'train_images');
    for i = 1:height(cohort)
        id = string(cohort.id_code(i));
        for ext = [".png",".jpg",".jpeg",".tif",".tiff"]
            candidate = fullfile(imageDir,id+ext);
            if isfile(candidate), imagePaths(i)=candidate; break; end
        end
        digest = '';
        if strlength(imagePaths(i))>0, digest = fileSHA256(imagePaths(i)); end
        expected(i)=struct('id_code',char(id),'true_grade',double(cohort.diagnosis(i)), ...
            'image_sha256',digest);
    end
    selectionDigest = '';
    if strlength(string(selectionPath))>0, selectionDigest=fileSHA256(selectionPath); end
    identity = struct('schema_version',3,'model_sha256',modelDigest, ...
        'config_sha256',configDigest,'config_json',configJson,'decision_policy',policy, ...
        'labels_sha256',fileSHA256(labelsPath), ...
        'selection_sha256',selectionDigest,'mode',char(mode), ...
        'preprocess_sha256',fileSHA256(fullfile(baseDir,'src','m3_grading','preprocessModelInput.m')), ...
        'source_sha256',sha256Bytes(unicode2native(jsonencode(sourceFiles),'UTF-8')), ...
        'source_files',sourceFiles,'expected_cohort',expected);
end

function policy=decisionPolicy(config)
    fields={'grade_thresholds','referable_head_threshold','confidence_temperature'};
    if ~isstruct(config) || ~isscalar(config) || ~all(isfield(config,fields))
        error('NetrAI:InvalidDecisionPolicy','Configuration must define all decision-policy values.');
    end
    thresholds=config.grade_thresholds;
    referral=config.referable_head_threshold;
    temperature=config.confidence_temperature;
    if ~isnumeric(thresholds) || ~isreal(thresholds) || ~isvector(thresholds) || ...
            numel(thresholds)~=4 || any(~isfinite(thresholds)) || ...
            any(thresholds<=0 | thresholds>=4) || any(diff(thresholds(:))<=0) || ...
            ~isnumeric(referral) || ~isreal(referral) || ~isscalar(referral) || ...
            ~isfinite(referral) || referral<=0 || referral>=1 || ...
            ~isnumeric(temperature) || ~isreal(temperature) || ~isscalar(temperature) || ...
            ~isfinite(temperature) || temperature<=0
        error('NetrAI:InvalidDecisionPolicy','Invalid ordinal, referral or calibration policy.');
    end
    policy=struct('grade_thresholds',double(thresholds(:)'), ...
        'referable_head_threshold',double(referral), ...
        'confidence_temperature',double(temperature));
end
