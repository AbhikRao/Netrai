function [heatmap, status] = generateGradCAM(net, modelInput, targetClass, recorded)
%GENERATEGRADCAM Attempt genuine Grad-CAM on the imported classifier.
% Returns [] with a diagnostic status when the imported ONNX graph cannot be
% differentiated by the installed MATLAB release. It never creates synthetic
% attention and therefore cannot mislabel lesion heuristics as Grad-CAM.
% recorded (optional): the ForwardRecorder of the grading call, so the referral map needs no second forward pass.

    if nargin < 4, recorded = []; end
    heatmap = [];
    status = "unavailable";
    if isempty(net)
        status = "network not loaded";
        return;
    end
    try
        if targetClass == 6
            % Referral logit of the legacy model or of the E_768 member ensemble: the same map as the Python pipeline (last
            % activation of each member, averaged over the members) in one forward pass. gradCAM's default feature layer would
            % explain only one E_768 member.
            heatmap = referralGradCAM(net, modelInput, recorded);
        else
            dlX = dlarray(single(modelInput), 'SSCB');
            heatmap = gradCAM(net, dlX, targetClass);
            heatmap = double(gather(extractdata(heatmap)));
            heatmap = squeeze(heatmap);
            heatmap = mat2gray(heatmap);
        end
        status = "available";
    catch ME
        status = "unavailable: " + string(ME.message);
        heatmap = [];
    end
end
