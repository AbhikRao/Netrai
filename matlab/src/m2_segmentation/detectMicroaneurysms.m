function [maMask, maCount, maCentroids] = detectMicroaneurysms(greenChannel, vesselMask, fovMask, odCenter, odRadius)
%DETECTMICROANEURYSMS Conservative candidate detector with FOV/OD exclusion.

    [h, w] = size(greenChannel);
    inverted = imcomplement(greenChannel);
    response = zeros(h, w);
    for r = [3 5 7]
        response = max(response, double(imtophat(inverted, strel('disk', r))));
    end
    filtered = imgaussfilt(response, 1.2);
    threshold = mean(filtered(:)) + 3.0 * std(filtered(:));
    candidates = filtered > threshold;
    if ~isempty(vesselMask)
        candidates(imdilate(vesselMask, strel('disk', 2))) = false;
    end
    if nargin >= 3 && ~isempty(fovMask)
        margin = max(3, round(min(h, w) * 0.02));
        safeFov = imerode(logical(fovMask), strel('disk', margin));
        candidates(~safeFov) = false;
    end
    if nargin >= 5 && ~isempty(odCenter) && odRadius > 0
        [xx, yy] = meshgrid(1:w, 1:h);
        candidates((xx-odCenter(1)).^2 + (yy-odCenter(2)).^2 <= (1.3*odRadius)^2) = false;
    end

    cc = bwconncomp(candidates);
    props = regionprops(cc, 'Area', 'Perimeter', 'Centroid', 'BoundingBox');
    minArea = max(5, round(0.00002 * h * w));
    maxArea = max(120, round(0.0005 * h * w));
    maMask = false(h, w);
    maCentroids = zeros(0, 2);
    for i = 1:cc.NumObjects
        area = props(i).Area;
        perimeter = props(i).Perimeter;
        box = props(i).BoundingBox;
        aspect = box(3) / max(box(4), 1);
        fillRatio = area / max(box(3) * box(4), 1);
        circularity = 0;
        if perimeter > 0
            circularity = 4*pi*area/(perimeter^2);
        end
        if area >= minArea && area <= maxArea && circularity > 0.65 ...
                && aspect >= 0.4 && aspect <= 2.5 && fillRatio >= 0.35
            maMask(cc.PixelIdxList{i}) = true;
            [~, peakOffset] = max(filtered(cc.PixelIdxList{i}));
            [peakY, peakX] = ind2sub([h,w], cc.PixelIdxList{i}(peakOffset));
            [refinedX, refinedY] = refineSubpixelPeak(filtered, peakX, peakY);
            maCentroids(end+1, :) = [refinedX, refinedY]; %#ok<AGROW>
        end
    end
    maCount = size(maCentroids, 1);
end
