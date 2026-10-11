function summary = aggregateValidationRuns(inputFolder, outputDir, varargin)
%AGGREGATEVALIDATIONRUNS Merge chunked MATLAB Online validation CSV files.
% Finds every per_image_results.csv below inputFolder, rejects duplicate IDs,
% recomputes all metrics, and writes one combined evidence bundle.
% ExpectedCohortCsv must contain id_code and diagnosis (or true_grade). When
% supplied, all expected cases and their labels must match exactly. Schema-2/3
% chunks already bind this cohort; schema-3 also freezes the decision policy.
% Historical schema-1/2 chunks do not establish policy-derived decision checks.
% Legacy chunks without an explicit cohort
% report coverage only over the observed chunks. Existing outputs are kept.

    if nargin < 2 || strlength(string(outputDir)) == 0
        outputDir = fullfile(inputFolder, 'combined');
    end
    inputFolder = string(inputFolder); outputDir = string(outputDir);
    addpath(fullfile(fileparts(mfilename('fullpath')),'src','utils'));
    assertAvailableOutput(outputDir);
    parser = inputParser;
    addParameter(parser,'ExpectedCohortCsv',"",@(x) ischar(x) || (isstring(x) && isscalar(x)));
    addParameter(parser,'ExpectedSelectionCsv',"",@(x) ischar(x) || (isstring(x) && isscalar(x)));
    parse(parser,varargin{:});
    expectedCsv = string(parser.Results.ExpectedCohortCsv);
    selectionCsv = string(parser.Results.ExpectedSelectionCsv);
    if strlength(expectedCsv)>0 && strlength(selectionCsv)>0
        error('NetrAI:InvalidExpectedCohort','Specify only one expected-cohort CSV option.');
    end
    if strlength(selectionCsv)>0, expectedCsv=selectionCsv; end
    files = dir(fullfile(inputFolder, '**', 'per_image_results.csv'));
    outputCanonical = string(java.io.File(char(outputDir)).getCanonicalPath());
    keep = true(numel(files),1);
    for k=1:numel(files)
        folderCanonical = string(java.io.File(files(k).folder).getCanonicalPath());
        keep(k) = folderCanonical ~= outputCanonical && ...
            ~contains(folderCanonical,string(filesep)+".netrai-staging-");
    end
    files = files(keep);
    if isempty(files)
        error('NetrAI:NoChunkFiles', 'No per_image_results.csv files found below %s.', inputFolder);
    end
    combined = table();
    identity = [];
    for i = 1:numel(files)
        identityPath=fullfile(files(i).folder,'run_identity.json');
        if ~isfile(identityPath)
            error('NetrAI:UnboundChunk','Missing run_identity.json; rerun legacy chunks before merging.');
        end
        try
            currentIdentity=jsondecode(fileread(identityPath));
        catch
            error('NetrAI:InvalidChunkIdentity','Chunk run identity is not valid JSON: %s.',identityPath);
        end
        validateIdentity(currentIdentity);
        if isempty(identity), identity=currentIdentity;
        elseif ~isequal(identity,currentIdentity)
            error('NetrAI:IncompatibleChunks','Model/config/cohort/mode identities differ.');
        end
        current = readValidationTable(fullfile(files(i).folder,files(i).name));
        current = validateRows(current,currentIdentity);
        combined = [combined; current]; %#ok<AGROW>
    end
    if any(ismissing(combined.id_code)) || numel(unique(combined.id_code)) ~= height(combined)
        error('NetrAI:DuplicateChunkIDs','Missing/duplicate IDs; do not merge overlapping chunks.');
    end
    cohortVerified = false;
    if identity.schema_version>=2
        verifyBoundCohort(combined,identity.expected_cohort);
        cohortVerified=true;
    end
    if strlength(expectedCsv)>0
        verifyExpectedCohort(combined,expectedCsv,identity);
        cohortVerified = true;
    end

    valid = combined.status == "success";
    yTrue = combined.true_grade(valid); yPred = combined.predicted_grade(valid);
    confMat = zeros(5,5);
    for i = 1:numel(yTrue)
        confMat(yTrue(i)+1,yPred(i)+1) = confMat(yTrue(i)+1,yPred(i)+1)+1;
    end
    addpath(genpath(fullfile(fileparts(mfilename('fullpath')), 'src')));
    if isempty(yTrue)
        classMetrics=struct('macroF1',NaN,'sensitivity',nan(1,5), ...
            'specificity',nan(1,5),'precision',nan(1,5),'f1Score',nan(1,5));
    else
        classMetrics = computeMetrics(confMat);
    end
    refTrue = yTrue>=2; refPred = yPred>=2;
    tp=nnz(refTrue&refPred); tn=nnz(~refTrue&~refPred);
    fp=nnz(~refTrue&refPred); fn=nnz(refTrue&~refPred);
    sensitivity=divide(tp,tp+fn); specificity=divide(tn,tn+fp);
    precision=divide(tp,tp+fp); refF1=divide(2*tp,2*tp+fp+fn);
    independentPred=logical(combined.referable_head_prediction(valid));
    itp=nnz(refTrue&independentPred); itn=nnz(~refTrue&~independentPred);
    ifp=nnz(~refTrue&independentPred); ifn=nnz(refTrue&~independentPred);
    disagreements=logical(combined.head_disagreement(valid)); concordant=~disagreements;
    summary = struct( ...
        'requested',height(combined),'successful',nnz(valid),'errors',nnz(combined.status=="error"), ...
        'recaptures',nnz(combined.status=="recapture"),'unresolved',nnz(~valid), ...
        'coverage',nnz(valid)/height(combined),'qwk',qwk(confMat), ...
        'expected_cohort_verified',cohortVerified, ...
        'decision_policy_verified',identity.schema_version==3, ...
        'accuracy',mean(yTrue==yPred),'mean_absolute_error',mean(abs(yTrue-yPred)), ...
        'within_one_grade_accuracy',mean(abs(yTrue-yPred)<=1),'macro_f1',classMetrics.macroF1, ...
        'referable_sensitivity',sensitivity,'referable_specificity',specificity, ...
        'referable_precision',precision,'referable_accuracy',(tp+tn)/numel(yTrue), ...
        'referable_f1',refF1,'referable_auc',auc(refTrue,combined.referable_probability(valid)), ...
        'referable_head_sensitivity',divide(itp,itp+ifn), ...
        'referable_head_specificity',divide(itn,itn+ifp), ...
        'referable_head_auc',auc(refTrue,combined.referable_head_probability(valid)), ...
        'head_disagreement_count',nnz(disagreements), ...
        'head_disagreement_rate',mean(disagreements), ...
        'concordant_coverage',mean(concordant), ...
        'concordant_sensitivity',divide(nnz(refTrue(concordant)&refPred(concordant)),nnz(refTrue(concordant))), ...
        'concordant_specificity',divide(nnz(~refTrue(concordant)&~refPred(concordant)),nnz(~refTrue(concordant))), ...
        'chunks_merged',numel(files));
    if cohortVerified, summary.coverage_scope="expected_cohort";
    else, summary.coverage_scope="observed_chunks"; end
    outputParent=fileparts(outputCanonical);
    if ~isfolder(outputParent), mkdir(outputParent); end
    stageDir=fullfile(outputParent,".netrai-staging-"+string(java.util.UUID.randomUUID));
    mkdir(stageDir);
    stageCleanup=onCleanup(@() removeStage(stageDir)); %#ok<NASGU>
    writetable(combined, fullfile(stageDir, 'per_image_results.csv'));
    fid=fopen(fullfile(stageDir,'run_identity.json'),'w');
    if fid<0, error('NetrAI:WriteFailed','Cannot write run identity'); end
    fprintf(fid,'%s\n',jsonencode(identity)); fclose(fid);
    writematrix(confMat, fullfile(stageDir, 'confusion_matrix.csv'));
    names=string(fieldnames(summary)); values=strings(numel(names),1);
    for i=1:numel(names), values(i)=string(summary.(names(i))); end
    writetable(table(names,values,'VariableNames',{'metric','value'}), ...
        fullfile(stageDir,'metrics_summary.csv'));
    fid=fopen(fullfile(stageDir,'aggregate_metrics.json'),'w');
    if fid<0, error('NetrAI:WriteFailed','Cannot create aggregate_metrics.json'); end
    fprintf(fid,'%s\n',jsonencode(summary,'PrettyPrint',true)); fclose(fid);
    grade=(0:4)'; support=sum(confMat,2); sensitivityGrade=classMetrics.sensitivity';
    specificityGrade=classMetrics.specificity'; precisionGrade=classMetrics.precision';
    f1Grade=classMetrics.f1Score';
    writetable(table(grade,support,sensitivityGrade,specificityGrade,precisionGrade,f1Grade), ...
        fullfile(stageDir,'per_grade_metrics.csv'));
    save(fullfile(stageDir,'validation_results.mat'),'summary','combined','confMat');
    assertAvailableOutput(outputDir);
    if isfolder(outputDir), rmdir(outputDir); end
    [published,message]=movefile(stageDir,outputDir);
    if ~published, error('NetrAI:WriteFailed','Cannot publish aggregate bundle: %s.',message); end
    fprintf('Merged %d unique images from %d chunks. QWK %.4f, errors %d.\n', ...
        height(combined),numel(files),summary.qwk,summary.errors);
end

function validateIdentity(identity)
    required = {'schema_version','model_sha256','config_sha256','labels_sha256', ...
        'mode','selection_sha256','preprocess_sha256'};
    if ~isstruct(identity) || ~isscalar(identity) || ...
            ~all(isfield(identity,required)) || ...
            ~isnumeric(identity.schema_version) || ~isscalar(identity.schema_version) || ...
            ~any(identity.schema_version==[1,2,3])
        error('NetrAI:InvalidChunkIdentity','Expected a supported bound validation identity; rerun unbound legacy chunks.');
    end
    for name = ["model_sha256","config_sha256","labels_sha256","preprocess_sha256"]
        if ~validDigest(identity.(name),false)
            error('NetrAI:InvalidChunkIdentity','Invalid SHA-256 identity field: %s.',name);
        end
    end
    if ~validDigest(identity.selection_sha256,true) || ...
            ~isTextScalar(identity.mode) || ~any(string(identity.mode)==["model","full"])
        error('NetrAI:InvalidChunkIdentity','Invalid selection digest or validation mode.');
    end
    if identity.schema_version>=2
        if ~isfield(identity,'source_sha256') || ~validDigest(identity.source_sha256,false) || ...
                ~isfield(identity,'expected_cohort') || ~isstruct(identity.expected_cohort) || ...
                isempty(identity.expected_cohort) || ...
                ~all(isfield(identity.expected_cohort,{'id_code','true_grade','image_sha256'}))
            error('NetrAI:InvalidChunkIdentity','Schema 2/3 must bind source content and the complete expected cohort.');
        end
        cases=identity.expected_cohort;
        for k=1:numel(cases)
            if ~isTextScalar(cases(k).id_code) || strlength(strtrim(string(cases(k).id_code)))==0 || ...
                    ~isscalar(cases(k).true_grade) || ...
                    ~validDigest(cases(k).image_sha256,true)
                error('NetrAI:InvalidChunkIdentity','Invalid expected-cohort record.');
            end
            checkGrades(cases(k).true_grade,'expected truth');
        end
        ids=string({cases.id_code});
        if numel(unique(ids))~=numel(ids)
            error('NetrAI:InvalidChunkIdentity','Expected-cohort IDs are duplicated.');
        end
    end
    if identity.schema_version==3
        validateFrozenPolicy(identity);
    end
end

function valid=validDigest(value,allowEmpty)
    valid=isTextScalar(value);
    if ~valid, return; end
    value=char(string(value));
    valid=(allowEmpty && isempty(value)) || ~isempty(regexp(value,'^[0-9a-fA-F]{64}$','once'));
end

function valid=isTextScalar(value)
    valid=(ischar(value) && (isrow(value) || isempty(value))) || ...
        (isstring(value) && isscalar(value) && ~ismissing(value));
end

function rows=validateRows(rows,identity)
    required={'id_code','true_grade','predicted_grade','referable_probability', ...
        'referable_head_probability','referable_head_prediction','head_disagreement','status'};
    if isempty(rows) || ~all(ismember(required,rows.Properties.VariableNames))
        error('NetrAI:InvalidChunk','Chunk CSV must contain nonempty validation rows and required columns.');
    end
    if identity.schema_version==3
        requiredPolicyColumns=["continuous_score","confidence", ...
            "calibrated_referable_probability","triage_action", ...
            "raw_prob_grade_"+string(0:4),"prob_grade_"+string(0:4)];
        if ~all(ismember(requiredPolicyColumns,string(rows.Properties.VariableNames)))
            error('NetrAI:InvalidChunkPolicy', ...
                'Schema-3 rows require the complete raw/calibrated probabilities and policy-derived outputs.');
        end
    end
    rows.id_code=string(rows.id_code); rows.status=string(rows.status);
    if any(ismissing(rows.id_code) | strlength(strtrim(rows.id_code))==0)
        error('NetrAI:DuplicateChunkIDs','Chunk IDs must be nonempty.');
    end
    if any(ismissing(rows.status) | ~ismember(rows.status,["success","error","recapture"]))
        error('NetrAI:InvalidChunkStatus','Only completed success/error/recapture rows can be aggregated.');
    end
    checkGrades(rows.true_grade,'true_grade');
    successful=rows.status=="success";
    checkGrades(rows.predicted_grade(successful),'predicted_grade');
    for name=["referable_probability","referable_head_probability"]
        checkProbability(rows.(name)(successful),name);
    end
    for name=["referable_head_prediction","head_disagreement"]
        values=rows.(name);
        if ~(isnumeric(values) || islogical(values)) || ...
                any(~isfinite(values) | (values~=0 & values~=1))
            error('NetrAI:InvalidChunkBoolean','%s must contain only finite 0/1 values.',name);
        end
        rows.(name)=logical(values);
    end
    disagreement=(rows.predicted_grade(successful)>=2) ~= rows.referable_head_prediction(successful);
    if any(rows.head_disagreement(successful)~=disagreement)
        error('NetrAI:InvalidChunkDecision','Head-disagreement flags conflict with the predicted decisions.');
    end
    % Older bound chunks may omit extra columns. Validate every optional
    % probability/vector column when present instead of trusting it silently.
    for prefix=["prob_grade_","raw_prob_grade_"]
        names=cellstr(prefix+string(0:4));
        present=ismember(names,rows.Properties.VariableNames);
        if any(present) && ~all(present)
            error('NetrAI:InvalidChunkProbability','Partial grade-probability vector.');
        end
        if all(present)
            values=rows{successful,names};
            checkProbability(values,prefix);
            sumTolerance=1e-6;
            if identity.schema_version==3, sumTolerance=1e-12; end
            if any(abs(sum(values,2)-1)>sumTolerance)
                error('NetrAI:InvalidChunkProbability','Grade probabilities must sum to one.');
            end
        end
    end
    for name=["confidence","calibrated_referable_probability","iqs"]
        if ismember(name,string(rows.Properties.VariableNames))
            checkProbability(rows.(name)(successful),name);
        end
    end
    for name=["continuous_score","elapsed_seconds"]
        if ismember(name,string(rows.Properties.VariableNames))
            values=rows.(name)(successful);
            if ~isnumeric(values) || any(~isfinite(values) | values<0) || ...
                    (name=="continuous_score" && any(values>4))
                error('NetrAI:InvalidChunkNumeric','Invalid successful-row values in %s.',name);
            end
        end
    end
    % writetable preserves about 15 significant decimal digits. This tolerance
    % only permits their serialization rounding, not different model scores.
    consistencyTolerance=1e-12;
    if any(successful)
        rawNames=cellstr("raw_prob_grade_"+string(0:4));
        if all(ismember(rawNames,rows.Properties.VariableNames))
            raw=rows{successful,rawNames};
            checkDerivedColumn(rows,successful,"referable_probability", ...
                sum(raw(:,3:5),2),consistencyTolerance);
            checkDerivedColumn(rows,successful,"continuous_score", ...
                raw*(0:4)',consistencyTolerance);
        end
        calibratedNames=cellstr("prob_grade_"+string(0:4));
        if all(ismember(calibratedNames,rows.Properties.VariableNames))
            calibrated=rows{successful,calibratedNames};
            checkDerivedColumn(rows,successful,"calibrated_referable_probability", ...
                sum(calibrated(:,3:5),2),consistencyTolerance);
            chosen=sub2ind(size(calibrated),(1:nnz(successful))', ...
                rows.predicted_grade(successful)+1);
            checkDerivedColumn(rows,successful,"confidence", ...
                calibrated(chosen),consistencyTolerance);
        end
    end
    if ismember('triage_action',rows.Properties.VariableNames)
        actual=string(rows.triage_action(successful));
        expected=repmat("auto_clear",nnz(successful),1);
        expected(rows.predicted_grade(successful)>=2)="refer";
        expected(disagreement)="uncertain_human_review";
        if any(ismissing(actual) | actual~=expected)
            error('NetrAI:InvalidChunkDecision','Triage actions conflict with the predicted decisions.');
        end
    end
    if identity.schema_version==3 && any(successful)
        verifyPolicyDecisions(rows,successful,identity.decision_policy,consistencyTolerance);
    end
end

function validateFrozenPolicy(identity)
    if ~isfield(identity,'decision_policy') || ~isfield(identity,'config_json') || ...
            ~isTextScalar(identity.config_json)
        error('NetrAI:InvalidChunkPolicy','Schema 3 requires a frozen policy and exact configuration JSON.');
    end
    % The raw UTF-8 configuration is preserved so policy values can be checked
    % against the captured config hash without loading a current deployment.
    bytes=unicode2native(char(string(identity.config_json)),'UTF-8');
    if ~strcmpi(sha256Bytes(bytes),char(string(identity.config_sha256)))
        error('NetrAI:InvalidChunkPolicy','Frozen configuration does not match config_sha256.');
    end
    try
        config=jsondecode(char(string(identity.config_json)));
    catch
        error('NetrAI:InvalidChunkPolicy','Frozen configuration must be valid JSON.');
    end
    policy=canonicalPolicy(identity.decision_policy,true);
    configuredPolicy=canonicalPolicy(config,false);
    if ~isequal(policy,configuredPolicy)
        error('NetrAI:ConflictingDecisionPolicy','Frozen policy differs from its content-bound configuration.');
    end
end

function policy=canonicalPolicy(value,strictFields)
    required={'grade_thresholds','referable_head_threshold','confidence_temperature'};
    if ~isstruct(value) || ~isscalar(value) || ~all(isfield(value,required)) || ...
            (strictFields && ~isequal(sort(fieldnames(value)),sort(required(:))))
        error('NetrAI:InvalidChunkPolicy','Decision policy must define exactly the supported three fields.');
    end
    thresholds=value.grade_thresholds;
    referral=value.referable_head_threshold;
    temperature=value.confidence_temperature;
    if ~isnumeric(thresholds) || ~isreal(thresholds) || ~isvector(thresholds) || ...
            numel(thresholds)~=4 || any(~isfinite(thresholds)) || ...
            any(thresholds<=0 | thresholds>=4) || any(diff(thresholds(:))<=0) || ...
            ~isnumeric(referral) || ~isreal(referral) || ~isscalar(referral) || ...
            ~isfinite(referral) || referral<=0 || referral>=1 || ...
            ~isnumeric(temperature) || ~isreal(temperature) || ~isscalar(temperature) || ...
            ~isfinite(temperature) || temperature<=0
        error('NetrAI:InvalidChunkPolicy','Invalid ordinal, referral or calibration policy.');
    end
    policy=struct('grade_thresholds',double(thresholds(:)'), ...
        'referable_head_threshold',double(referral), ...
        'confidence_temperature',double(temperature));
end

function verifyPolicyDecisions(rows,successful,policy,tolerance)
    thresholds=double(policy.grade_thresholds(:)');
    expectedGrade=sum(rows.continuous_score(successful)>=thresholds,2);
    expectedHead=rows.referable_head_probability(successful)>=policy.referable_head_threshold;
    if any(rows.predicted_grade(successful)~=expectedGrade) || ...
            any(rows.referable_head_prediction(successful)~=expectedHead)
        error('NetrAI:InconsistentChunkDecision', ...
            'Predicted grade or referral-head decision conflicts with the frozen policy.');
    end
    raw=rows{successful,cellstr("raw_prob_grade_"+string(0:4))};
    scaled=log(raw+1e-10)/policy.confidence_temperature;
    scaled=scaled-max(scaled,[],2);
    expectedCalibration=exp(scaled);
    expectedCalibration=expectedCalibration./sum(expectedCalibration,2);
    calibrated=rows{successful,cellstr("prob_grade_"+string(0:4))};
    if any(abs(calibrated-expectedCalibration)>tolerance,'all')
        error('NetrAI:InconsistentChunkCalibration', ...
            'Calibrated probabilities conflict with the frozen confidence temperature.');
    end
end

function checkDerivedColumn(rows,successful,name,expected,tolerance)
    if ismember(name,string(rows.Properties.VariableNames)) && ...
            any(abs(rows.(name)(successful)-expected)>tolerance)
        error('NetrAI:InconsistentChunkResult', ...
            '%s conflicts with the stored grade-probability vector.',name);
    end
end

function verifyBoundCohort(rows,cases)
    if ~ismember('image_sha256',rows.Properties.VariableNames)
        error('NetrAI:UnboundChunk','Schema-2 chunk rows must include image_sha256.');
    end
    ids=reshape(string({cases.id_code}),[],1);
    truth=reshape([cases.true_grade],[],1);
    verifyCaseMapping(rows,ids,truth);
    [~,positions]=ismember(rows.id_code,ids);
    hashes=reshape(string({cases.image_sha256}),[],1);
    boundHashes=hashes(positions);
    if any(rows.status=="success" & strlength(boundHashes)==0)
        error('NetrAI:UnboundSuccessfulImage','Successful results require a bound source-image SHA-256.');
    end
    if ismember('image_sha256',rows.Properties.VariableNames)
        actual=string(rows.image_sha256);
        % Empty CSV fields are imported as missing strings. They are only
        % legitimate for an unresolved case whose source file was absent.
        actual(ismissing(actual))="";
        if any(lower(actual)~=lower(boundHashes))
            error('NetrAI:ConflictingImageIdentity','Chunk image digests differ from the bound cohort.');
        end
    end
end

function assertAvailableOutput(path)
    if isfile(path)
        error('NetrAI:OutputExists','Output destination already exists: %s.',path);
    end
    if isfolder(path)
        contents=dir(path);
        contents=contents(~ismember({contents.name},{'.','..'}));
        if ~isempty(contents)
            error('NetrAI:OutputExists','Use a fresh output directory; existing results are preserved: %s.',path);
        end
    end
end

function removeStage(path)
    if isfolder(path), rmdir(path,'s'); end
end

function checkGrades(values,name)
    if ~isnumeric(values) || any(~isfinite(values) | values~=floor(values) | values<0 | values>4)
        error('NetrAI:InvalidGrade','%s must contain finite integer grades 0-4.',name);
    end
end

function checkProbability(values,name)
    if ~isnumeric(values) || any(~isfinite(values) | values<0 | values>1,'all')
        error('NetrAI:InvalidChunkProbability','%s must contain finite probabilities in [0,1].',name);
    end
end

function verifyExpectedCohort(rows,path,identity)
    if ~isfile(path), error('NetrAI:ExpectedCohortMissing','Expected cohort CSV not found: %s.',path); end
    expected=readValidationTable(path);
    if ~ismember('id_code',expected.Properties.VariableNames)
        error('NetrAI:InvalidExpectedCohort','Expected cohort CSV must include id_code and diagnosis/true_grade.');
    end
    if ismember('diagnosis',expected.Properties.VariableNames), truth=expected.diagnosis;
    elseif ismember('true_grade',expected.Properties.VariableNames), truth=expected.true_grade;
    else, error('NetrAI:InvalidExpectedCohort','Expected cohort CSV must include diagnosis or true_grade.'); end
    ids=string(expected.id_code);
    if isempty(ids) || any(ismissing(ids) | strlength(strtrim(ids))==0) || numel(unique(ids))~=numel(ids)
        error('NetrAI:InvalidExpectedCohort','Expected cohort IDs must be nonempty and unique.');
    end
    checkGrades(truth,'expected truth');
    verifyCaseMapping(rows,ids,truth);
    if strlength(string(identity.selection_sha256))>0
        addpath(fullfile(fileparts(mfilename('fullpath')),'src','utils'));
        if ~strcmpi(fileSHA256(path),identity.selection_sha256)
            error('NetrAI:IncompatibleExpectedCohort','Expected cohort CSV does not match the bound selection CSV.');
        end
    end
end

function rows=readValidationTable(path)
    options=detectImportOptions(path,'TextType','string','VariableNamingRule','preserve');
    textColumns=intersect({'id_code','status','triage_action','error','image_sha256'},options.VariableNames);
    if ~isempty(textColumns), options=setvartype(options,textColumns,'string'); end
    rows=readtable(path,options);
end

function verifyCaseMapping(rows,ids,truth)
    [found,positions]=ismember(ids,rows.id_code);
    if height(rows)~=numel(ids) || any(~found)
        error('NetrAI:IncompleteCohort','Observed chunk IDs do not exactly cover the expected cohort.');
    end
    if any(truth~=rows.true_grade(positions))
        error('NetrAI:ConflictingCohortLabels','Chunk truth labels differ from the expected cohort.');
    end
end

function value=divide(a,b)
    if b==0, value=NaN; else, value=a/b; end
end

function value=qwk(matrix)
    n=size(matrix,1); total=sum(matrix,'all');
    expected=sum(matrix,2)*sum(matrix,1)/total;
    [r,c]=ndgrid(0:n-1,0:n-1); weights=((r-c).^2)/(n-1)^2;
    value=1-sum(weights.*matrix,'all')/sum(weights.*expected,'all');
end

function value=auc(labels,scores)
    labels=logical(labels(:)); scores=scores(:); p=nnz(labels); n=nnz(~labels);
    if p==0 || n==0, value=NaN; return; end
    [sorted,order]=sort(scores,'ascend'); ranks=(1:numel(scores))'; first=1;
    while first<=numel(scores)
        last=first;
        while last<numel(scores) && sorted(last+1)==sorted(first), last=last+1; end
        ranks(first:last)=mean(first:last); first=last+1;
    end
    original=zeros(size(ranks)); original(order)=ranks;
    value=(sum(original(labels))-p*(p+1)/2)/(p*n);
end
