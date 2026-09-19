function summary = aggregateValidationRuns(inputFolder, outputDir)
%AGGREGATEVALIDATIONRUNS Merge chunked MATLAB Online validation CSV files.
% Finds every per_image_results.csv below inputFolder, removes duplicate IDs,
% recomputes all metrics, and writes one combined evidence bundle.

    if nargin < 2 || strlength(string(outputDir)) == 0
        outputDir = fullfile(inputFolder, 'combined');
    end
    inputFolder = string(inputFolder); outputDir = string(outputDir);
    files = dir(fullfile(inputFolder, '**', 'per_image_results.csv'));
    if isempty(files)
        error('NetrAI:NoChunkFiles', 'No per_image_results.csv files found below %s.', inputFolder);
    end
    combined = table();
    for i = 1:numel(files)
        current = readtable(fullfile(files(i).folder, files(i).name), ...
            'TextType','string', 'VariableNamingRule','preserve');
        combined = [combined; current]; %#ok<AGROW>
    end
    required = {'id_code','true_grade','predicted_grade','referable_probability', ...
        'referable_head_probability','referable_head_prediction', ...
        'head_disagreement','status'};
    if ~all(ismember(required, combined.Properties.VariableNames))
        error('NetrAI:InvalidChunk', 'Chunk CSV files do not have the expected validation columns.');
    end
    [~, uniqueRows] = unique(combined.id_code, 'stable');
    combined = combined(sort(uniqueRows), :);
    if ~isfolder(outputDir), mkdir(outputDir); end
    writetable(combined, fullfile(outputDir, 'per_image_results.csv'));

    valid = combined.status == "success" & isfinite(combined.predicted_grade);
    yTrue = combined.true_grade(valid); yPred = combined.predicted_grade(valid);
    if isempty(yTrue), error('NetrAI:NoValidResults', 'No successful rows were found.'); end
    confMat = zeros(5,5);
    for i = 1:numel(yTrue)
        confMat(yTrue(i)+1,yPred(i)+1) = confMat(yTrue(i)+1,yPred(i)+1)+1;
    end
    writematrix(confMat, fullfile(outputDir, 'confusion_matrix.csv'));
    addpath(genpath(fullfile(fileparts(mfilename('fullpath')), 'src')));
    classMetrics = computeMetrics(confMat);
    refTrue = yTrue>=2; refPred = yPred>=2;
    tp=nnz(refTrue&refPred); tn=nnz(~refTrue&~refPred);
    fp=nnz(~refTrue&refPred); fn=nnz(refTrue&~refPred);
    sensitivity=divide(tp,tp+fn); specificity=divide(tn,tn+fp);
    precision=divide(tp,tp+fp); refF1=divide(2*precision*sensitivity,precision+sensitivity);
    independentPred=logical(combined.referable_head_prediction(valid));
    itp=nnz(refTrue&independentPred); itn=nnz(~refTrue&~independentPred);
    ifp=nnz(~refTrue&independentPred); ifn=nnz(refTrue&~independentPred);
    disagreements=logical(combined.head_disagreement(valid)); concordant=~disagreements;
    summary = struct( ...
        'requested',height(combined),'successful',nnz(valid),'errors',nnz(~valid), ...
        'coverage',nnz(valid)/height(combined),'qwk',qwk(confMat), ...
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
    names=string(fieldnames(summary)); values=strings(numel(names),1);
    for i=1:numel(names), values(i)=string(summary.(names(i))); end
    writetable(table(names,values,'VariableNames',{'metric','value'}), ...
        fullfile(outputDir,'metrics_summary.csv'));
    fid=fopen(fullfile(outputDir,'aggregate_metrics.json'),'w');
    if fid<0, error('NetrAI:WriteFailed','Cannot create aggregate_metrics.json'); end
    fprintf(fid,'%s\n',jsonencode(summary,'PrettyPrint',true)); fclose(fid);
    grade=(0:4)'; support=sum(confMat,2); sensitivityGrade=classMetrics.sensitivity';
    specificityGrade=classMetrics.specificity'; precisionGrade=classMetrics.precision';
    f1Grade=classMetrics.f1Score';
    writetable(table(grade,support,sensitivityGrade,specificityGrade,precisionGrade,f1Grade), ...
        fullfile(outputDir,'per_grade_metrics.csv'));
    save(fullfile(outputDir,'validation_results.mat'),'summary','combined','confMat');
    fprintf('Merged %d unique images from %d chunks. QWK %.4f, errors %d.\n', ...
        height(combined),numel(files),summary.qwk,summary.errors);
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
