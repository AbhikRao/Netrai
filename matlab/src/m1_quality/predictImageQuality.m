function [qualityClass, probabilities, logits] = predictImageQuality(net, modelInput, config)
%PREDICTIMAGEQUALITY Run the learned candidate without integrating its decision.

    if size(modelInput,1) ~= config.input_size || ...
            size(modelInput,2) ~= config.input_size || size(modelInput,3) ~= 3
        error('NetrAI:InvalidQualityInput', 'Unexpected quality model input size.');
    end
    raw = predict(net, dlarray(single(modelInput), 'SSCB'));
    logits = double(gather(extractdata(raw)));
    logits = logits(:)';
    if numel(logits) ~= 3 || any(~isfinite(logits))
        error('NetrAI:InvalidQualityOutput', 'Expected three finite logits.');
    end
    shifted = logits ./ config.confidence_temperature;
    shifted = shifted - max(shifted);
    probabilities = exp(shifted) ./ sum(exp(shifted));
    [~, qualityClass] = max(probabilities(1:2));
    qualityClass = qualityClass - 1;
    if probabilities(3) >= config.reject_threshold
        qualityClass = 2;
    end
end
