function featureVec = extractClinicalFeatures(segResults)
%EXTRACTCLINICALFEATURES Summarize retinal candidates in a 12-D feature vector.
% These features support explanations; the exported Fold-0 grade head itself
% was trained from images and does not consume this vector.

    featureVec = zeros(1, 12);
    [rows, cols] = size(segResults.vesselMask);
    if isfield(segResults, 'odCenter') && ~isempty(segResults.odCenter)
        center = segResults.odCenter;
    else
        center = [cols/2 rows/2];
    end

    if isfield(segResults, 'maCentroids') && ~isempty(segResults.maCentroids)
        points = segResults.maCentroids;
        counts = [ ...
            sum(points(:,1)>=center(1) & points(:,2)>=center(2)), ...
            sum(points(:,1)< center(1) & points(:,2)>=center(2)), ...
            sum(points(:,1)< center(1) & points(:,2)< center(2)), ...
            sum(points(:,1)>=center(1) & points(:,2)< center(2))];
        featureVec(1:4) = min(counts/50, 1);
    elseif isfield(segResults, 'maCount')
        featureVec(1:4) = min(segResults.maCount/4/50, 1);
    end

    if isfield(segResults, 'hemMask') && ~isempty(segResults.hemMask)
        mask = logical(segResults.hemMask);
        r = floor(rows/2); c = floor(cols/2);
        valid = true(size(mask));
        if isfield(segResults,'fovMask'), valid = logical(segResults.fovMask); end
        regions = {mask(r+1:end,c+1:end), mask(r+1:end,1:c), ...
                   mask(1:r,1:c), mask(1:r,c+1:end)};
        fovs = {valid(r+1:end,c+1:end), valid(r+1:end,1:c), ...
                valid(1:r,1:c), valid(1:r,c+1:end)};
        for i = 1:4
            featureVec(4+i) = nnz(regions{i} & fovs{i}) / max(nnz(fovs{i}), 1);
        end
    end

    if isfield(segResults, 'foveaDist') && segResults.foveaDist >= 0
        featureVec(9) = min(segResults.foveaDist/5, 1);
    else
        featureVec(9) = 1;
    end
    if isfield(segResults, 'softCount')
        featureVec(10) = double(segResults.softCount > 0);
    end
    if isfield(segResults, 'fovMask') && any(segResults.fovMask(:))
        denominator = nnz(segResults.fovMask);
    else
        denominator = rows*cols;
    end
    featureVec(11) = nnz(segResults.vesselMask) / max(denominator, 1);
    if isfield(segResults, 'nvDetected')
        featureVec(12) = double(segResults.nvDetected);
    end
    featureVec(~isfinite(featureVec)) = 0;
    featureVec = min(max(featureVec, 0), 1);
end
