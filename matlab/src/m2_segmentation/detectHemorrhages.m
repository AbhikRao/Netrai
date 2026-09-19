function [hemMask, hemCount, hemStats] = detectHemorrhages(greenChannel, vesselMask, maMask, fovMask, odCenter, odRadius)
%DETECTHEMORRHAGES Candidate detector with vessel, FOV, and OD exclusions.

    [h, w] = size(greenChannel);
    inverted = imcomplement(greenChannel);
    response = zeros(h, w);
    for r = [10 15]
        response = max(response, double(imtophat(inverted, strel('disk', r))));
    end
    if max(response(:)) > 0
        candidates = imbinarize(mat2gray(response), graythresh(mat2gray(response)));
    else
        candidates = false(h, w);
    end
    if ~isempty(vesselMask)
        candidates(imdilate(vesselMask, strel('disk', 3))) = false;
    end
    if ~isempty(maMask)
        candidates(maMask) = false;
    end
    if nargin >= 4 && ~isempty(fovMask)
        margin = max(3, round(min(h, w) * 0.02));
        safeFov = imerode(logical(fovMask), strel('disk', margin));
        candidates(~safeFov) = false;
    end
    if nargin >= 6 && ~isempty(odCenter) && odRadius > 0
        [xx, yy] = meshgrid(1:w, 1:h);
        candidates((xx-odCenter(1)).^2 + (yy-odCenter(2)).^2 <= (1.3*odRadius)^2) = false;
    end

    cc = bwconncomp(candidates);
    props = regionprops(cc, 'Area', 'Perimeter');
    minArea = max(50, round(0.0003 * h * w));
    maxArea = max(5000, round(0.02 * h * w));
    hemMask = false(h, w);
    dotCount = 0; blotCount = 0;
    for i = 1:cc.NumObjects
        area = props(i).Area;
        if area < minArea || area > maxArea
            continue;
        end
        hemMask(cc.PixelIdxList{i}) = true;
        circularity = 0;
        if props(i).Perimeter > 0
            circularity = 4*pi*area/(props(i).Perimeter^2);
        end
        if circularity > 0.6
            dotCount = dotCount + 1;
        else
            blotCount = blotCount + 1;
        end
    end
    hemCount = dotCount + blotCount;
    hemStats = struct('dot_count', dotCount, 'blot_count', blotCount);
end
