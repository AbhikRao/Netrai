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
    [bundle, modelDigest] = verifyDeploymentBundle(modelPath, configPath);
    verifyRuntimeDeploymentSources(bundle);
    if exist('importNetworkFromONNX', 'file') ~= 2
        error('NetrAI:ONNXImporterMissing', [ ...
            'importNetworkFromONNX is unavailable. Use MATLAB R2023b or newer ' ...
            'and install the free Deep Learning Toolbox Converter for ONNX ' ...
            'Model Format support package from Add-Ons.']);
    end

    config = jsondecode(fileread(configPath));
    required = {'input_size', 'imagenet_mean', 'imagenet_std', ...
        'grade_thresholds', 'confidence_temperature', ...
        'referable_head_threshold', 'head_disagreement_action', ...
        'preprocessing_version', 'black_crop_threshold', 'ben_graham_sigma'};
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
    if string(config.preprocessing_version) ~= "legacy-area-v1" || ...
            ~isscalar(config.black_crop_threshold) || config.black_crop_threshold ~= 7 || ...
            ~isscalar(config.ben_graham_sigma) || config.ben_graham_sigma ~= 10
        error('NetrAI:InvalidConfig','Unsupported released preprocessing contract.');
    end

    if ~isscalar(config.input_size) || config.input_size < 256 || config.input_size > 1024 || ...
            mod(config.input_size, 32) ~= 0 || ...
            numel(config.imagenet_mean) ~= 3 || any(~isfinite(config.imagenet_mean)) || ...
            numel(config.imagenet_std) ~= 3 || ...
            any(~isfinite(config.imagenet_std) | config.imagenet_std <= 0) || ...
            numel(config.grade_thresholds) ~= 4 || ...
            any(~isfinite(config.grade_thresholds)) || any(diff(config.grade_thresholds) <= 0)
        error('NetrAI:InvalidConfig', 'Invalid input normalization or ordinal thresholds.');
    end

    persistent cachedNet cachedModelPath cachedDigest
    if isempty(cachedNet) || cachedModelPath ~= modelPath || ~strcmp(cachedDigest,modelDigest)
        try
            importedNet = importNetworkFromONNX(modelPath, ...
                'InputDataFormats','BCSS', 'OutputDataFormats','BC');
        catch ME
            error('NetrAI:ONNXImportFailed', ...
                'Could not import %s. Install the ONNX converter support package. Original error: %s', ...
                modelPath, ME.message);
        end
        if ~strcmp(fileSHA256(modelPath),modelDigest)
            error('NetrAI:BundleIntegrity', 'Model changed during ONNX import.');
        end
        % Import may take minutes; source resolution must still match before
        % the imported network can enter the reusable deployment cache.
        verifyRuntimeDeploymentSources(bundle);
        cachedNet = importedNet;
        cachedModelPath = modelPath;
        cachedDigest = modelDigest;
    end
    net = cachedNet;
    config.deployment_identity = struct('bundle_schema_version',bundle.schema_version, ...
        'model_sha256',modelDigest, ...
        'config_sha256',bundle.artifacts(strcmp({bundle.artifacts.role},'config')).sha256, ...
        'preprocessing_version',config.preprocessing_version);
end
