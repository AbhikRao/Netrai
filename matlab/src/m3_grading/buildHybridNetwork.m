function network = buildHybridNetwork(varargin) %#ok<STOUT,INUSD>
%BUILDHYBRIDNETWORK Retired non-equivalent architecture experiment.
% Use loadNetrAIModel to load the actual validated EfficientNet-B4 ONNX model.

    error('NetrAI:RetiredArchitecture', [ ...
        'The former MATLAB graph was not the trained NetrAI architecture. ' ...
        'Call loadNetrAIModel with models/netrai_fold0.onnx instead.']);
end
