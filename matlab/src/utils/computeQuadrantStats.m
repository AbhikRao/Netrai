function stats = computeQuadrantStats(binaryMask, center)
%COMPUTEQUADRANTSTATS Compute lesion statistics per retinal quadrant.
%   stats = computeQuadrantStats(binaryMask, center)
%
%   Divides the image into 4 quadrants around 'center' and counts
%   lesion pixels and connected components in each.
%
%   Inputs:
%       binaryMask - Binary lesion mask
%       center     - [x, y] center point (typically fovea or OD)
%   Output:
%       stats - struct with fields:
%           counts    - [ST, SN, IT, IN] lesion component counts
%           areas     - [ST, SN, IT, IN] total lesion pixel areas
%           totalCount - total connected components
%           totalArea  - total lesion pixels

    [h, w] = size(binaryMask);
    cx = round(center(1));
    cy = round(center(2));

    % Clamp center to image bounds
    cx = max(1, min(w, cx));
    cy = max(1, min(h, cy));

    % Define quadrant masks
    % ST = Superior-Temporal, SN = Superior-Nasal
    % IT = Inferior-Temporal, IN = Inferior-Nasal
    quadMasks = cell(1, 4);
    quadMasks{1} = false(h, w); quadMasks{1}(1:cy, cx:end) = true;    % ST
    quadMasks{2} = false(h, w); quadMasks{2}(1:cy, 1:cx) = true;      % SN
    quadMasks{3} = false(h, w); quadMasks{3}(cy:end, cx:end) = true;   % IT
    quadMasks{4} = false(h, w); quadMasks{4}(cy:end, 1:cx) = true;     % IN

    quadNames = {'ST', 'SN', 'IT', 'IN'};
    counts = zeros(1, 4);
    areas  = zeros(1, 4);

    for q = 1:4
        regionMask = binaryMask & quadMasks{q};
        cc = bwconncomp(regionMask);
        counts(q) = cc.NumObjects;
        areas(q) = sum(regionMask(:));
    end

    stats.counts     = counts;
    stats.areas      = areas;
    stats.quadNames  = quadNames;
    stats.totalCount = sum(counts);
    stats.totalArea  = sum(areas);
end
