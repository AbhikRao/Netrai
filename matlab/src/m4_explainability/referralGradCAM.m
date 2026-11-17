function [heatmap, check, memberMaps, coarse] = referralGradCAM(net, modelInput, recorded)
%REFERRALGRADCAM Grad-CAM of the referral logit of a NetrAI network: the legacy model (one member) or a mean ensemble (E_768),
% averaged over the members.
%
% Each member ends in conv_head -> SiLU (A) -> global average pooling (p) -> head (Linear-BN-SiLU-Linear-BN-SiLU) -> referral Linear.
% The pooled vector p is the spatial mean of A, so the Grad-CAM weight of channel c is proportional to dy/dp_c, computed in closed form
% through the eval-mode head. Member k's map is ReLU(sum_c dy/dp_c * A_c), min-max scaled and resized (bilinear) to the input; the
% maps are averaged and scaled again - the definition of the Python pipeline (modules/explainability.py head_gradcam, equal to
% pytorch_grad_cam at each member's bn2) and of the X11 alignment study. One forward pass, no automatic differentiation through the
% backbone. check.max_logit_error compares the recomputed referral logit with the network output; above 1e-3 the head was not
% identified correctly and an error is raised (no substitute map).
%
% recorded (optional): a ForwardRecorder that already ran this network on this input (the grading call in main_pipeline). Its
% member activations and output are used instead of a second forward pass; the map is the same (check.forward_passes is then 0).
    useRecorded = nargin >= 3 && ~isempty(recorded) && recorded.hasForward();
    if useRecorded
        if ~isequal(recorded.Input, single(modelInput))
            error('NetrAI:GradCAMRecordedInputMismatch', 'The recorded forward pass belongs to another input.');
        end
        members = recorded.Members;
        activations = recorded.Activations;
        output = recorded.Output;
    else
        members = referralHeadMembers(net);
        dlX = dlarray(single(modelInput), 'SSCB');
        activations = cell(1, numel(members));
        [activations{:}, output] = predict(net, dlX, 'Outputs', [[members.activation] string(net.OutputNames{1})]);
    end
    side = size(modelInput, 1);
    maps = zeros(side, side, numel(members));
    logits = zeros(1, numel(members));
    coarse = cell(1, numel(members));                                % member maps at the feature-map resolution (tests)
    for k = 1:numel(members)
        A = double(gather(extractdata(activations{k})));            % S x S x C
        [logits(k), dydp] = referralGradient(members(k).head, squeeze(mean(A, [1 2])));
        cam = max(sum(A .* reshape(dydp, 1, 1, []), 3), 0);
        coarse{k} = cam;
        maps(:, :, k) = imresize(rescale01(cam), [side side], 'bilinear');   % scale, then resize (pytorch_grad_cam order)
    end
    ensembleLogit = double(gather(extractdata(output)));
    check = struct('members', numel(members), 'max_logit_error', abs(mean(logits) - ensembleLogit(6)), ...
        'forward_passes', double(~useRecorded));
    if check.max_logit_error > 1e-3
        error('NetrAI:GradCAMHeadMismatch', 'Recomputed referral logit differs from the network output by %.2e.', check.max_logit_error);
    end
    heatmap = rescale01(mean(max(maps, 0), 3));
    memberMaps = maps;
end

function [y, dydp] = referralGradient(h, p)
% Eval-mode head: z = BN(W x + b), a = SiLU(z); y = w_ref a2 + b_ref; returns y and dy/dp.
    [z1, s1] = bn(h.bn1, double(h.fc1.Weights) * p + double(h.fc1.Bias(:)));
    a1 = silu(z1);
    [z2, s2] = bn(h.bn2, double(h.fc2.Weights) * a1 + double(h.fc2.Bias(:)));
    y = double(h.ref.Weights) * silu(z2) + double(h.ref.Bias);
    dz2 = double(h.ref.Weights(:)) .* dsilu(z2);
    dz1 = (double(h.fc2.Weights)' * (dz2 .* s2)) .* dsilu(z1);
    dydp = double(h.fc1.Weights)' * (dz1 .* s1);
end

function [z, scale] = bn(layer, u)
    scale = double(layer.Scale(:)) ./ sqrt(double(layer.TrainedVariance(:)) + double(layer.Epsilon));
    z = (u - double(layer.TrainedMean(:))) .* scale + double(layer.Offset(:));
end

function a = silu(z)
    a = z ./ (1 + exp(-z));
end

function d = dsilu(z)
    s = 1 ./ (1 + exp(-z));
    d = s .* (1 + z .* (1 - s));
end

function x = rescale01(x)
    x = x - min(x(:));
    x = x / (max(x(:)) + 1e-7);
end
