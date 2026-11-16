classdef (InferiorClasses = {?dlarray}) ForwardRecorder < handle
%FORWARDRECORDER Runs a NetrAI network once and keeps what Grad-CAM needs from that same forward pass.
%
% main_pipeline hands this object to predictDRGrade in place of the network. When the grading code calls predict, the real network
% runs on the given input and returns its six outputs exactly as before; the last activation of each member is kept as well, so
% referralGradCAM does not have to run the network a second time. Nothing is precomputed or substituted: the grade comes from the
% forward pass that predictDRGrade itself started. If the activations cannot be requested, the plain forward pass is used and
% Grad-CAM later runs its own (hasForward stays false).
    properties (SetAccess = private)
        Net
        Members
        Input = []
        Activations = {}
        Output = []
        ForwardPasses = 0
    end
    methods
        function obj = ForwardRecorder(net)
            obj.Net = net;
            obj.Members = referralHeadMembers(net);      % errors for a network without the NetrAI head
        end

        function output = predict(obj, dlX)
            acts = cell(1, numel(obj.Members));
            obj.Activations = {}; obj.Output = []; obj.Input = [];
            try
                [acts{:}, output] = predict(obj.Net, dlX, 'Outputs', [[obj.Members.activation] string(obj.Net.OutputNames{1})]);
            catch
                output = predict(obj.Net, dlX);          % grading must not depend on the extra outputs
                obj.ForwardPasses = obj.ForwardPasses + 1;
                return;
            end
            obj.ForwardPasses = obj.ForwardPasses + 1;
            obj.Input = single(gather(extractdata(dlX)));
            obj.Activations = acts;
            obj.Output = output;
        end

        function tf = hasForward(obj)
            tf = ~isempty(obj.Activations) && ~isempty(obj.Output);
        end
    end
end
