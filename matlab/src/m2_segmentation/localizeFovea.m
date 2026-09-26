function foveaCenter = localizeFovea(greenChannel, odCenter, odRadius)
%LOCALIZEFOVEA Localizes the fovea based on the optic disc location.
%   foveaCenter = localizeFovea(greenChannel, odCenter, odRadius)
%   Inputs:
%       greenChannel - Green channel of the image
%       odCenter - [x, y] center of the optic disc
%       odRadius - Radius of the optic disc
%   Outputs:
%       foveaCenter - [x, y] coordinates of the fovea center

    if isempty(greenChannel)
        error('Green channel is empty.');
    end
    
    [h, w] = size(greenChannel);
    
    % Determine temporal direction (OD is usually nasal)
    if odCenter(1) < w/2
        % OD is in left half, fovea is to the right
        dir = 1; 
    else
        % OD is in right half, fovea is to the left
        dir = -1;
    end
    
    % Define search region center: ~2.5 OD diameters (or 5 radii) temporal
    searchCenterX = odCenter(1) + dir * (5 * odRadius);
    searchCenterY = odCenter(2);
    
    % Smooth green channel
    smoothed = imgaussfilt(greenChannel, 10);
    
    % Define search window limits
    windowSize = 2 * odRadius;
    xMin = max(1, round(searchCenterX - windowSize));
    xMax = min(w, round(searchCenterX + windowSize));
    yMin = max(1, round(searchCenterY - windowSize));
    yMax = min(h, round(searchCenterY + windowSize));
    
    % Find darkest point in the search region
    if xMin < xMax && yMin < yMax
        searchRegion = double(smoothed(yMin:yMax, xMin:xMax));
        valid = greenChannel(yMin:yMax, xMin:xMax) > 10;
        if ~any(valid,'all'), foveaCenter = []; return; end
        searchRegion(~valid) = Inf;
        [~, minIdx] = min(searchRegion(:));
        [minY, minX] = ind2sub(size(searchRegion), minIdx);
        
        foveaCenter = [xMin + minX - 1, yMin + minY - 1];
    else
        % Fallback
        foveaCenter = [];
    end
end
