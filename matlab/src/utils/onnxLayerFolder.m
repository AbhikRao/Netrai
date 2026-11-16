function layerRoot = onnxLayerFolder()
%ONNXLAYERFOLDER Folder of the generated ONNX custom-layer namespaces (matlab/onnx_layers, git-ignored), on the path.
% A network imported by importOnnxNetwork runs only while this folder is on the path, so code that reuses a cached network calls
% this before each use: the path can change after the import (a test runner, for example, restores it after its tests).
    layerRoot = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'onnx_layers');
    if ~isfolder(layerRoot)
        mkdir(layerRoot);
    end
    if ~any(strcmp(strsplit(path, pathsep), layerRoot))
        addpath(layerRoot);
    end
end
