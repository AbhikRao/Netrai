function [trainedNet, info] = trainDRClassifier(varargin) %#ok<STOUT,INUSD>
%TRAINDRCLASSIFIER Training is intentionally not duplicated in MATLAB.
% The authoritative, reproducible training workflow is
% research/legacy_training.ipynb (EfficientNet-B4, 512 px, focal loss, Fold 0).
% Earlier code in this file built a different 224 px multi-input network and
% swallowed training failures, so it could not reproduce the deployed model.

    error('NetrAI:UseAuthoritativeTrainingNotebook', [ ...
        'MATLAB retraining is not packaged because it would not reproduce the ' ...
        'validated checkpoint. Use research/legacy_training.ipynb for training, then ' ...
        'run python/export_matlab_model.py to refresh the MATLAB ONNX artifact.']);
end
