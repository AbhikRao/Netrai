function [grade, probabilities, isReferable, continuousScore, logits, ...
        referableProbability, referableHeadPositive, headDisagreement, ...
        referableLogit] = predictDRGrade(net, modelInput, thresholds, referableThreshold)
%PREDICTDRGRADE Run the exported trained model and threshold its score.
% No rule-based or fabricated fallback is used. Prediction errors propagate
% to the caller so a failed model cannot be mistaken for a clinical result.

    if isempty(net)
        error('NetrAI:MissingNetwork', 'A loaded ONNX network is required.');
    end
    if nargin < 4
        referableThreshold = 0.5;
    end
    if ~isscalar(referableThreshold) || ~isfinite(referableThreshold) || ...
            referableThreshold <= 0 || referableThreshold >= 1
        error('NetrAI:InvalidReferableThreshold', ...
            'Referable-head threshold must be strictly between zero and one.');
    end
    thresholds = sort(double(thresholds(:)'));
    if numel(thresholds) ~= 4 || any(~isfinite(thresholds))
        error('NetrAI:InvalidThresholds', 'Exactly four finite grade thresholds are required.');
    end
    if size(modelInput, 1) ~= 512 || size(modelInput, 2) ~= 512 || size(modelInput, 3) ~= 3
        error('NetrAI:InvalidModelInput', 'Model input must be 512-by-512-by-3.');
    end

    dlX = dlarray(single(modelInput), 'SSCB');
    rawOutput = predict(net, dlX);
    modelOutputs = double(gather(extractdata(rawOutput)));
    modelOutputs = modelOutputs(:)';
    if numel(modelOutputs) ~= 6 || any(~isfinite(modelOutputs))
        error('NetrAI:InvalidModelOutput', ...
            'Expected five grade logits plus one referable logit from ONNX model.');
    end
    logits = modelOutputs(1:5);
    referableLogit = modelOutputs(6);

    shifted = logits - max(logits);
    probabilities = exp(shifted);
    probabilities = probabilities ./ sum(probabilities);
    continuousScore = probabilities * (0:4)';
    grade = sum(continuousScore > thresholds);
    grade = min(max(grade, 0), 4);
    isReferable = grade >= 2;
    if referableLogit >= 0
        referableProbability = 1 / (1 + exp(-referableLogit));
    else
        expLogit = exp(referableLogit);
        referableProbability = expLogit / (1 + expLogit);
    end
    referableHeadPositive = referableProbability >= referableThreshold;
    headDisagreement = logical(isReferable) ~= logical(referableHeadPositive);
end
