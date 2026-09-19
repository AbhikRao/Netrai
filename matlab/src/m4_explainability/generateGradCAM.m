function [heatmap, status] = generateGradCAM(net, modelInput, targetClass)
%GENERATEGRADCAM Attempt genuine Grad-CAM on the imported classifier.
% Returns [] with a diagnostic status when the imported ONNX graph cannot be
% differentiated by the installed MATLAB release. It never creates synthetic
% attention and therefore cannot mislabel lesion heuristics as Grad-CAM.

    heatmap = [];
    status = "unavailable";
    if isempty(net)
        status = "network not loaded";
        return;
    end
    try
        dlX = dlarray(single(modelInput), 'SSCB');
        heatmap = gradCAM(net, dlX, targetClass);
        heatmap = double(gather(extractdata(heatmap)));
        heatmap = squeeze(heatmap);
        heatmap = mat2gray(heatmap);
        status = "available";
    catch ME
        status = "unavailable: " + string(ME.message);
        heatmap = [];
    end
end
