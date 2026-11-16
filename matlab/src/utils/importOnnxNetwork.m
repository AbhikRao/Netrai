function net = importOnnxNetwork(modelPath, namespace, varargin)
%IMPORTONNXNETWORK importNetworkFromONNX with its generated custom-layer namespace in one fixed folder kept on the path.
% importNetworkFromONNX writes +<namespace> into the current folder, so a network imported there stopped working once that
% folder left the path or the current folder changed ("Execution failed during layers ..."). This imports inside
% matlab/onnx_layers (git-ignored, see onnxLayerFolder); callers reusing a cached network call onnxLayerFolder before each use.
    info = dir(modelPath);
    if numel(info) ~= 1 || info.isdir
        error('NetrAI:MissingArtifact', 'ONNX file not found: %s', modelPath);
    end
    modelPath = fullfile(info.folder, info.name);       % absolute: the import runs in another folder
    layerRoot = onnxLayerFolder();
    previous = cd(layerRoot);
    restore = onCleanup(@() cd(previous));
    net = importNetworkFromONNX(modelPath, varargin{:}, 'Namespace', namespace);
end
