function [nvMask, nvDetected] = detectNeovascularization(greenChannel, vesselMask, odCenter, odRadius, fovMask) %#ok<INUSD>
%DETECTNEOVASCULARIZATION Conservative clustered vessel-density screen.

    [h, w] = size(vesselMask);
    nvMask = false(h, w);
    nvDetected = false;
    if isempty(vesselMask) || isempty(odCenter)
        return;
    end
    blockSize = max(16, round(min(h, w)/16));
    nRows = ceil(h/blockSize); nCols = ceil(w/blockSize);
    densities = zeros(nRows, nCols);
    valid = false(nRows, nCols);
    [xx, yy] = meshgrid(1:w, 1:h);
    odExclude = (xx-odCenter(1)).^2 + (yy-odCenter(2)).^2 <= (1.5*odRadius)^2;
    if nargin >= 5 && ~isempty(fovMask)
        margin = max(5, round(min(h, w)*0.04));
        safeFov = imerode(logical(fovMask), strel('disk', margin));
    else
        safeFov = true(h, w);
    end
    for row = 1:nRows
        y1 = (row-1)*blockSize+1; y2 = min(row*blockSize, h);
        for col = 1:nCols
            x1 = (col-1)*blockSize+1; x2 = min(col*blockSize, w);
            area = (y2-y1+1)*(x2-x1+1);
            if nnz(safeFov(y1:y2,x1:x2))/area < 0.90 || ...
                    nnz(odExclude(y1:y2,x1:x2))/area > 0.25
                continue;
            end
            valid(row,col) = true;
            densities(row,col) = nnz(vesselMask(y1:y2,x1:x2))/area;
        end
    end
    values = densities(valid);
    if isempty(values), return; end
    candidates = densities > max(0.30, 3*mean(values)) & valid;
    cc = bwconncomp(candidates, 8);
    props = regionprops(cc, 'Area');
    for i = 1:cc.NumObjects
        if props(i).Area < 8, continue; end
        nvDetected = true;
        [blockRows, blockCols] = ind2sub([nRows nCols], cc.PixelIdxList{i});
        for j = 1:numel(blockRows)
            y1 = (blockRows(j)-1)*blockSize+1; y2 = min(blockRows(j)*blockSize,h);
            x1 = (blockCols(j)-1)*blockSize+1; x2 = min(blockCols(j)*blockSize,w);
            nvMask(y1:y2,x1:x2) = true;
        end
    end
end
