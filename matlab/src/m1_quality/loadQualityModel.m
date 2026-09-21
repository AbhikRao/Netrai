function [net, config] = loadQualityModel(modelPath, configPath)
%LOADQUALITYMODEL Load the evaluated, non-deployed EyeQ candidate.

    if ~isfile(modelPath) || ~isfile(configPath)
        error('NetrAI:QualityArtifactMissing', 'Quality model/config is missing.');
    end
    config = jsondecode(fileread(configPath));
    required = {'input_size','imagenet_mean','imagenet_std','classes', ...
        'reject_threshold','confidence_temperature','deployment_status'};
    for index = 1:numel(required)
        if ~isfield(config, required{index})
            error('NetrAI:InvalidQualityConfig', ...
                'Missing quality config field: %s', required{index});
        end
    end
    if string(config.deployment_status) ~= "research_candidate_not_deployed"
        error('NetrAI:InvalidQualityStatus', 'Unexpected quality candidate status.');
    end
    if config.reject_threshold <= 0 || config.reject_threshold >= 1
        error('NetrAI:InvalidQualityThreshold', ...
            'Reject threshold must be strictly between zero and one.');
    end
    if config.confidence_temperature <= 0 || ~isfinite(config.confidence_temperature)
        error('NetrAI:InvalidQualityTemperature', ...
            'Quality confidence temperature must be finite and positive.');
    end
    net = importNetworkFromONNX(modelPath, ...
        'InputDataFormats','BCSS', 'OutputDataFormats','BC');
end
