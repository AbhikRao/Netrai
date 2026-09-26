%CALIBRATECONFIDENCE Apply fitted scalar temperature to classification probabilities.
%
% Syntax: [calProbs, temperature] = calibrateConfidence(rawProbs, temperature)
%
% Inputs:
%   rawProbs    - [1x5] vector of raw probabilities
%   temperature - (Optional) fitted temperature (identity default: 1.0)
%
% Outputs:
%   calProbs    - [1x5] vector of calibrated probabilities
%   temperature - Temperature used for scaling
function [calProbs, temperature] = calibrateConfidence(rawProbs, temperature)
    if nargin < 2 || isnan(temperature)
        temperature = 1.0;
    end
    if ~isscalar(temperature) || ~isfinite(temperature) || temperature <= 0
        error('NetrAI:InvalidTemperature', ...
            'Temperature must be a finite positive scalar.');
    end
    
    % Convert probs to logits
    logits = log(rawProbs + 1e-10);
    
    % Apply temperature scaling
    scaled = logits / temperature;
    
    % Softmax
    scaled = scaled - max(scaled);
    calProbs = exp(scaled) / sum(exp(scaled));
end
