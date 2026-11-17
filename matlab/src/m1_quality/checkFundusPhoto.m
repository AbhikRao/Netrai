function guard = checkFundusPhoto(rawImg, paramsPath)
%CHECKFUNDUSPHOTO Is the input a fundus photograph? Guard score of the quality candidate's 576-D embedding (X15).
% Mahalanobis distance of the global-average-pooled features of the frozen ImageNet MobileNetV3-Small encoder inside the quality
% candidate (full-frame 224-pixel input) to fundus images of the datasets that trained the published weights - the same score as
% python/modules/fundus_guard.py from the same parameters file (models/fundus_guard.json + .f32, SHA-256 checked). Above
% guard.threshold the image is not a fundus photograph (guard.is_fundus = false); guard.action ("reject" or "flag") says what the
% pipeline does then. The network and parameters are loaded once per session.
    persistent cache
    if nargin < 2 || strlength(string(paramsPath)) == 0
        paramsPath = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'models', 'fundus_guard.json');
    end
    paramsPath = string(paramsPath);
    if isempty(cache) || cache.paramsPath ~= paramsPath
        cache = loadGuard(paramsPath);
    end
    onnxLayerFolder();      % the cached network's custom layers must be on the path (it may have changed since the import)
    modelInput = preprocessQualityModelInput(rawImg, cache.config);
    feature = predict(cache.net, dlarray(single(modelInput), 'SSCB'), 'Outputs', cache.layer);
    embedding = double(gather(extractdata(feature)));
    centred = embedding(:) - cache.mean;
    score = sqrt(max(centred' * cache.precision * centred, 0));
    guard = struct('is_fundus', score <= cache.threshold, 'score', score, 'threshold', cache.threshold, 'action', cache.action);
end

function cache = loadGuard(paramsPath)
    if ~isfile(paramsPath)
        error('NetrAI:FundusGuardMissing', 'Fundus guard parameters not found: %s', paramsPath);
    end
    meta = jsondecode(fileread(paramsPath));
    folder = fileparts(paramsPath);
    binary = fullfile(folder, meta.parameters_file);
    if string(fileSHA256(binary)) ~= string(meta.parameters_sha256)
        error('NetrAI:FundusGuardIntegrity', '%s does not match the SHA-256 in %s.', meta.parameters_file, paramsPath);
    end
    if ~any(string(meta.action) == ["reject" "flag"])
        error('NetrAI:FundusGuardIntegrity', 'The guard parameters do not name an action (reject or flag).');
    end
    [net, config] = loadQualityModel(fullfile(folder, 'quality_model_candidate.onnx'), fullfile(folder, 'quality_model_candidate.json'));
    if string(config.checkpoint_sha256) ~= string(meta.encoder_checkpoint_sha256)
        error('NetrAI:FundusGuardIntegrity', 'The quality candidate is not the encoder the guard was fitted with.');
    end
    fid = fopen(binary, 'r');
    values = fread(fid, inf, 'float32=>double', 0, 'ieee-le');
    fclose(fid);
    d = double(meta.dimension);
    if numel(values) ~= d + d * d
        error('NetrAI:FundusGuardIntegrity', 'Unexpected guard parameter size.');
    end
    pools = find(arrayfun(@(l) isa(l, 'nnet.cnn.layer.GlobalAveragePooling2DLayer'), net.Layers));
    cache = struct('paramsPath', paramsPath, 'net', net, 'config', config, 'layer', string(net.Layers(pools(end)).Name), ...
        'mean', values(1:d), 'precision', reshape(values(d+1:end), d, d)', 'threshold', double(meta.threshold), ...
        'action', string(meta.action));
end
