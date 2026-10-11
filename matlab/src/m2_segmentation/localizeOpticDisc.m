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
    hasSuppliedFov = nargin >= 3 && ~isempty(fovMask);
    if hasSuppliedFov
        if ~(isnumeric(fovMask) || islogical(fovMask)) || ~isreal(fovMask) ...
                || ~ismatrix(fovMask) || ~isequal(size(fovMask), [h w]) ...
                || any(~isfinite(fovMask(:))) || any(fovMask(:) < 0)
            error('NetrAI:InvalidFOV', 'FOV must be a finite nonnegative 2-D mask matching the image.');
        end
        validFov = logical(fovMask);
        if ~any(validFov(:))
            error('NetrAI:EmptyFOV', 'Optic-disc localization requires a nonempty retinal FOV.');
        end
    else
        validFov = red > 10;
    end
    brightness = imgaussfilt(red, max(6, 0.03 * minDim));
    brightness(~validFov) = 0;
    [peakValue, peakIndex] = max(brightness(:));
    if peakValue <= 0
        error('NetrAI:OpticDiscUnavailable', 'No supported brightness peak is available for optic-disc localization.');
    end
    [peakY, peakX] = ind2sub([h, w], peakIndex);

    blurred = imgaussfilt(red, 2);
    [centers, radii] = imfindcircles(blurred, [minRadius maxRadius], ...
        'ObjectPolarity', 'bright', 'Sensitivity', 0.90);
    bestScore = -Inf;
    bestIndex = 0;
    [xx, yy] = meshgrid(1:w, 1:h);
    for i = 1:size(centers, 1)
        centerX = round(centers(i, 1)); centerY = round(centers(i, 2));
        if centerX < 1 || centerX > w || centerY < 1 || centerY > h ...
                || ~validFov(centerY, centerX)
            continue;
        end
        if norm(centers(i, :) - [peakX, peakY]) > 2 * maxRadius
            continue;
        end
        candidate = (xx - centers(i, 1)).^2 + (yy - centers(i, 2)).^2 <= radii(i)^2;
        pixels = double(red(candidate & validFov));
        if ~isempty(pixels) && mean(pixels) > bestScore
            bestScore = mean(pixels);
            bestIndex = i;
        end
    end
    if bestIndex > 0
        odCenter = centers(bestIndex, :);
        odRadius = radii(bestIndex);
    else
        odCenter = [peakX, peakY];
        odRadius = round(0.075 * minDim);
    end
end
