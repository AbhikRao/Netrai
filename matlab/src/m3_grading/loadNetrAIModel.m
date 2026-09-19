function [net, config] = loadNetrAIModel(modelPath, configPath)
%LOADNETRAIMODEL Load the exported NetrAI ONNX model and deployment config.
% Requires MATLAB R2023b or newer and the free Deep Learning Toolbox
% Converter for ONNX Model Format support package.

    arguments
        modelPath (1,1) string
        configPath (1,1) string
    end

    if ~isfile(modelPath)
        error('NetrAI:ModelNotFound', 'ONNX model not found: %s', modelPath);
    end
    if ~isfile(configPath)
        error('NetrAI:ConfigNotFound', 'Deployment config not found: %s', configPath);
    end
    if exist('importNetworkFromONNX', 'file') ~= 2
        error('NetrAI:ONNXImporterMissing', [ ...
            'importNetworkFromONNX is unavailable. Use MATLAB R2023b or newer ' ...
            'and install the free Deep Learning Toolbox Converter for ONNX ' ...
            'Model Format support package from Add-Ons.']);
    end

    config = jsondecode(fileread(configPath));
    required = {'input_size', 'imagenet_mean', 'imagenet_std', ...
        'grade_thresholds', 'confidence_temperature', ...
        'referable_head_threshold', 'head_disagreement_action'};
    for i = 1:numel(required)
        if ~isfield(config, required{i})
            error('NetrAI:InvalidConfig', 'Missing config field: %s', required{i});
        end
    end
    if ~isscalar(config.confidence_temperature) || ...
            ~isfinite(config.confidence_temperature) || config.confidence_temperature <= 0
        error('NetrAI:InvalidConfig', ...
            'confidence_temperature must be a finite positive scalar.');
    end
    if ~isscalar(config.referable_head_threshold) || ...
            ~isfinite(config.referable_head_threshold) || ...
            config.referable_head_threshold <= 0 || config.referable_head_threshold >= 1
        error('NetrAI:InvalidConfig', ...
            'referable_head_threshold must be strictly between zero and one.');
    end
    if string(config.head_disagreement_action) ~= "uncertain_human_review"
        error('NetrAI:UnsafeConfig', ...
            'Head disagreement must route to uncertain_human_review.');
    end

    persistent cachedNet cachedModelPath cachedStamp
    modelInfo = dir(modelPath);
    stamp = [modelInfo.datenum, modelInfo.bytes]; %#ok<DATNM>
    if isempty(cachedNet) || cachedModelPath ~= modelPath || ~isequal(cachedStamp, stamp)
        try
            cachedNet = importNetworkFromONNX(modelPath, ...
                'InputDataFormats','BCSS', 'OutputDataFormats','BC');
        catch ME
            error('NetrAI:ONNXImportFailed', ...
                'Could not import %s. Install the ONNX converter support package. Original error: %s', ...
                modelPath, ME.message);
        end
        cachedModelPath = modelPath;
        cachedStamp = stamp;
    end
    net = cachedNet;
end
