function [odCenter, odRadius] = localizeOpticDisc(img, greenChannel, fovMask) %#ok<INUSD>
%LOCALIZEOPTICDISC Brightness-guided optic-disc localization.

    if isempty(img) || ndims(img) ~= 3
        error('NetrAI:InvalidImage', 'A non-empty RGB image is required.');
    end
    [h, w, ~] = size(img);
    minDim = min(h, w);
    minRadius = max(8, round(0.04 * minDim));
    maxRadius = max(minRadius + 1, round(0.12 * minDim));
    red = img(:, :, 1);
    brightness = imgaussfilt(red, max(6, 0.03 * minDim));
    if nargin >= 3 && ~isempty(fovMask)
        brightness(~fovMask) = 0;
    else
        brightness(red <= 10) = 0;
    end
    [peakValue, peakIndex] = max(brightness(:));
    [peakY, peakX] = ind2sub([h, w], peakIndex);

    blurred = imgaussfilt(red, 2);
    [centers, radii] = imfindcircles(blurred, [minRadius maxRadius], ...
        'ObjectPolarity', 'bright', 'Sensitivity', 0.90);
    bestScore = -Inf;
    bestIndex = 0;
    [xx, yy] = meshgrid(1:w, 1:h);
    for i = 1:size(centers, 1)
        if norm(centers(i, :) - [peakX, peakY]) > 2 * maxRadius
            continue;
        end
        candidate = (xx - centers(i, 1)).^2 + (yy - centers(i, 2)).^2 <= radii(i)^2;
        pixels = double(red(candidate));
        if ~isempty(pixels) && mean(pixels) > bestScore
            bestScore = mean(pixels);
            bestIndex = i;
        end
    end
    if bestIndex > 0
        odCenter = centers(bestIndex, :);
        odRadius = radii(bestIndex);
    elseif peakValue > 0
        odCenter = [peakX, peakY];
        odRadius = round(0.075 * minDim);
    else
        odCenter = [0.75 * w, 0.5 * h];
        odRadius = round(0.08 * minDim);
    end
end
