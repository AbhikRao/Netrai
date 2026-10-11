function [manifest, modelDigest] = verifyDeploymentBundle(modelPath, configPath, bundlePath)
%VERIFYDEPLOYMENTBUNDLE Reject mixed model, policy and inference-source artifacts.
    arguments
        modelPath (1,1) string
        configPath (1,1) string
        bundlePath (1,1) string = ""
    end
    base = fileparts(configPath);
    if strlength(bundlePath) == 0
        bundlePath = fullfile(base, 'netrai_bundle.json');
    end
    if ~isfile(bundlePath)
        error('NetrAI:BundleMissing', 'Missing deployment bundle: %s', bundlePath);
    end
    try
        manifest = jsondecode(fileread(bundlePath));
    catch
        error('NetrAI:InvalidBundle', 'Deployment bundle must be valid JSON.');
    end
    roles = ["model", "config", "preprocessing", "grading", "calibration", "routing", "decoding"];
    [~, modelName, modelExtension] = fileparts(modelPath);
    [~, configName, configExtension] = fileparts(configPath);
    paths = [modelName+modelExtension, configName+configExtension, ...
        "../src/m3_grading/preprocessModelInput.m", ...
        "../src/m3_grading/predictDRGrade.m", ...
        "../src/m4_explainability/calibrateConfidence.m", ...
        "../src/utils/assignScreeningTriage.m", ...
        "../src/utils/readScreeningImage.m"];
    if ~isstruct(manifest) || ~isscalar(manifest) || ...
            ~isfield(manifest,'schema_version') || ...
            ~isequal(manifest.schema_version,2) || ~isfield(manifest,'artifacts') || ...
            ~isstruct(manifest.artifacts) || numel(manifest.artifacts) ~= numel(roles) || ...
            ~all(isfield(manifest.artifacts,{'role','path','sha256'}))
        error('NetrAI:InvalidBundle', 'Unsupported deployment bundle schema.');
    end
    actualRoles = string({manifest.artifacts.role});
    if ~isequal(sort(actualRoles),sort(roles))
        error('NetrAI:InvalidBundle', 'Bundle must bind each of the seven deployment roles once.');
    end
    if canonical(fullfile(base,paths(1))) ~= canonical(modelPath)
        error('NetrAI:InvalidBundle', 'Model and config must share the bundle directory.');
    end
    modelDigest = '';
    for i = 1:numel(roles)
        item = manifest.artifacts(actualRoles == roles(i));
        if ~isscalar(string(item.path)) || string(item.path) ~= paths(i) || ...
                ~ischar(item.sha256) || isempty(regexp(item.sha256,'^[0-9a-f]{64}$','once'))
            error('NetrAI:InvalidBundle', 'Invalid path or SHA-256 for role %s.',roles(i));
        end
        path = fullfile(base,paths(i));
        if ~isfile(path) || ~strcmp(fileSHA256(path),item.sha256)
            error('NetrAI:BundleIntegrity', 'Deployment artifact changed: %s.',roles(i));
        end
        if roles(i) == "model", modelDigest = item.sha256; end
    end
end

function path = canonical(path)
    path = string(java.io.File(char(path)).getCanonicalPath());
end
