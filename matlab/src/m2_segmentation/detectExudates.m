function [exMask, hardCount, softCount, nearestFoveaDist] = detectExudates(img, greenChannel, foveaCenter, odCenter, odRadius, fovMask)
%DETECTEXUDATES Detect locally bright lesion candidates within the safe FOV.

    [h, w] = size(greenChannel);
    lab = rgb2lab(img);
    L = lab(:, :, 1);                 % MATLAB CIE L*: 0..100
    yellow = lab(:, :, 3);            % positive b* is yellow
    [xx, yy] = meshgrid(1:w, 1:h);
    odMask = (xx-odCenter(1)).^2 + (yy-odCenter(2)).^2 <= (1.3*odRadius)^2;
    if nargin >= 6 && ~isempty(fovMask)
        margin = max(3, round(min(h, w) * 0.03));
        safeFov = imerode(logical(fovMask), strel('disk', margin));
    else
        safeFov = true(h, w);
    end
    valid = safeFov & ~odMask;

    localBackground = imgaussfilt(L, max(3, 0.02*min(h, w)));
    localContrast = L - localBackground;
    values = localContrast(valid);
    if isempty(values)
        contrastThreshold = Inf;
    else
        contrastThreshold = mean(values) + 3.5*std(values);
    end
    hardMask = localContrast > contrastThreshold & L > 51 & yellow > 5 & valid;
    hardMask = bwareaopen(imopen(hardMask, strel('disk', 1)), ...
        max(5, round(0.00002*h*w)));
    cc = bwconncomp(hardMask);
    props = regionprops(cc, 'Area', 'Centroid');
    cleanHard = false(h, w);
    hardCentroids = zeros(0, 2);
    maxArea = max(1000, round(0.004*h*w));
    for i = 1:cc.NumObjects
        if props(i).Area <= maxArea
            cleanHard(cc.PixelIdxList{i}) = true;
            hardCentroids(end+1, :) = props(i).Centroid; %#ok<AGROW>
        end
    end
    hardCount = size(hardCentroids, 1);

    [gx, gy] = imgradientxy(im2double(greenChannel));
    gradientMagnitude = hypot(gx, gy);
    gradThreshold = mean(gradientMagnitude(:)) + std(gradientMagnitude(:));
    softMask = L > 63 & yellow < 8 & gradientMagnitude < gradThreshold & valid;
    softMask = bwareaopen(imopen(softMask, strel('disk', 2)), 10);
    softComponents = bwconncomp(softMask);
    softCount = softComponents.NumObjects;
    exMask = cleanHard | softMask;

    nearestFoveaDist = -1;
    if ~isempty(hardCentroids) && odRadius > 0
        distances = vecnorm(hardCentroids - foveaCenter, 2, 2);
        nearestFoveaDist = min(distances) / odRadius;
    end
end
