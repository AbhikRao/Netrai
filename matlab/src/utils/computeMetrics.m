% COMPUTEMETRICS Computes classification metrics from a confusion matrix
%
%   metrics = computeMetrics(confMat) computes sensitivity, specificity, 
%   precision, F1-score and overall accuracy from an NxN confusion matrix.
%   Rows are assumed to be true labels, columns are predicted labels.
%
%   Input:
%       confMat - NxN numeric matrix representing the confusion matrix
%
%   Output:
%       metrics - struct containing the following fields:
%           .accuracy    - overall accuracy
%           .sensitivity - 1xN array of per-class sensitivity (recall)
%           .specificity - 1xN array of per-class specificity
%           .precision   - 1xN array of per-class precision
%           .f1Score     - 1xN array of per-class F1-score
%           .macroF1     - macro-averaged F1-score

function metrics = computeMetrics(confMat)
    
    % Input validation
    if ~ismatrix(confMat) || size(confMat, 1) ~= size(confMat, 2)
        error('Input must be a square confusion matrix.');
    end
    
    N = size(confMat, 1);
    
    % Overall accuracy
    totalSamples = sum(confMat, 'all');
    if totalSamples == 0
        error('Confusion matrix is empty (sum is 0).');
    end
    
    correctPredictions = sum(diag(confMat));
    metrics.accuracy = correctPredictions / totalSamples;
    
    % Initialize per-class metrics
    metrics.sensitivity = zeros(1, N);
    metrics.specificity = zeros(1, N);
    metrics.precision = zeros(1, N);
    metrics.f1Score = zeros(1, N);
    
    % Compute metrics for each class
    for i = 1:N
        TP = confMat(i, i);
        FN = sum(confMat(i, :)) - TP;
        FP = sum(confMat(:, i)) - TP;
        TN = totalSamples - (TP + FN + FP);
        
        % Sensitivity (Recall / True Positive Rate)
        if (TP + FN) > 0
            metrics.sensitivity(i) = TP / (TP + FN);
        else
            metrics.sensitivity(i) = NaN;
        end
        
        % Specificity (True Negative Rate)
        if (TN + FP) > 0
            metrics.specificity(i) = TN / (TN + FP);
        else
            metrics.specificity(i) = NaN;
        end
        
        % Precision (Positive Predictive Value)
        if (TP + FP) > 0
            metrics.precision(i) = TP / (TP + FP);
        else
            metrics.precision(i) = NaN;
        end
        
        % F1-Score
        if ~isnan(metrics.precision(i)) && ~isnan(metrics.sensitivity(i)) && (metrics.precision(i) + metrics.sensitivity(i)) > 0
            metrics.f1Score(i) = 2 * (metrics.precision(i) * metrics.sensitivity(i)) / (metrics.precision(i) + metrics.sensitivity(i));
        else
            metrics.f1Score(i) = NaN;
        end
    end
    
    % Macro F1 (ignoring NaNs)
    metrics.macroF1 = mean(metrics.f1Score(~isnan(metrics.f1Score)));
end
