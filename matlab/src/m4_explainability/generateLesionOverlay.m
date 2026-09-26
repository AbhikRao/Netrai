function overlayImg = generateLesionOverlay(originalImg, segResults)
%GENERATELESIONOVERLAY Create a color-coded candidate lesion overlay on fundus image.
%
%   overlayImg = generateLesionOverlay(originalImg, segResults)
%
%   Inputs:
%       originalImg - Original fundus image [HxWx3 uint8]
%       segResults  - Struct containing lesion segmentation results
%
%   Outputs:
%       overlayImg  - Image with color-coded lesions and anatomical landmarks

    overlayImg = originalImg;
    [h, w, ~] = size(originalImg);
    
    % 1. Microaneurysms: Red circles
    maPts = [];
    if isfield(segResults, 'maCentroids') && ~isempty(segResults.maCentroids)
        maPts = segResults.maCentroids;
    elseif isfield(segResults, 'MAs') && isfield(segResults.MAs, 'centroids')
        maPts = segResults.MAs.centroids;
    end
    if ~isempty(maPts)
        try
            circles = [maPts, repmat(4, size(maPts, 1), 1)];
            overlayImg = insertShape(overlayImg, 'FilledCircle', circles, 'Color', 'red', 'Opacity', 0.8);
        catch
            % Manual drawing fallback
            for k = 1:size(maPts, 1)
                cx = round(maPts(k, 1)); cy = round(maPts(k, 2));
                rRange = max(1, cy-3):min(h, cy+3);
                cRange = max(1, cx-3):min(w, cx+3);
                overlayImg(rRange, cRange, 1) = 255;
                overlayImg(rRange, cRange, 2:3) = 0;
            end
        end
    end
    
    % 2. Hemorrhages: Dark red tint
    hemM = [];
    if isfield(segResults, 'hemMask') && any(segResults.hemMask(:))
        hemM = segResults.hemMask;
    elseif isfield(segResults, 'hemorrhages') && isfield(segResults.hemorrhages, 'mask')
        hemM = segResults.hemorrhages.mask;
    end
    if ~isempty(hemM)
        hemM = imresize(hemM, [h, w], 'nearest');
        R = overlayImg(:,:,1); G = overlayImg(:,:,2); B = overlayImg(:,:,3);
        R(hemM) = uint8(double(R(hemM))*0.4 + 200*0.6);
        G(hemM) = uint8(double(G(hemM))*0.3);
        B(hemM) = uint8(double(B(hemM))*0.3);
        overlayImg(:,:,1) = R; overlayImg(:,:,2) = G; overlayImg(:,:,3) = B;
    end
    
    % 3. Hard Exudates: Bright Yellow tint
    exM = [];
    if isfield(segResults, 'exMask') && any(segResults.exMask(:))
        exM = segResults.exMask;
    elseif isfield(segResults, 'hardExudates') && isfield(segResults.hardExudates, 'mask')
        exM = segResults.hardExudates.mask;
    end
    if ~isempty(exM)
        exM = imresize(exM, [h, w], 'nearest');
        R = overlayImg(:,:,1); G = overlayImg(:,:,2); B = overlayImg(:,:,3);
        R(exM) = uint8(double(R(exM))*0.3 + 255*0.7);
        G(exM) = uint8(double(G(exM))*0.3 + 255*0.7);
        B(exM) = uint8(double(B(exM))*0.2);
        overlayImg(:,:,1) = R; overlayImg(:,:,2) = G; overlayImg(:,:,3) = B;
    end
    
    % 4. Optic Disc: Green circle outline
    odC = []; odR = 40;
    if isfield(segResults, 'odCenter') && ~isempty(segResults.odCenter)
        odC = segResults.odCenter;
        if isfield(segResults, 'odRadius') && ~isempty(segResults.odRadius), odR = segResults.odRadius; end
    elseif isfield(segResults, 'opticDisc') && isfield(segResults.opticDisc, 'centroid')
        odC = segResults.opticDisc.centroid;
        if isfield(segResults.opticDisc, 'radius'), odR = segResults.opticDisc.radius; end
    end
    if ~isempty(odC)
        try
            overlayImg = insertShape(overlayImg, 'Circle', [odC, odR], 'Color', 'green', 'LineWidth', 3);
        catch
            % fallback
        end
    end
    
    % 5. Fovea: Cyan / Blue crosshair
    fovC = [];
    if isfield(segResults, 'foveaCenter') && ~isempty(segResults.foveaCenter)
        fovC = segResults.foveaCenter;
    elseif isfield(segResults, 'fovea') && isfield(segResults.fovea, 'centroid')
        fovC = segResults.fovea.centroid;
    end
    if ~isempty(fovC)
        cx = round(fovC(1)); cy = round(fovC(2));
        try
            lines = [cx-12, cy, cx+12, cy; cx, cy-12, cx, cy+12];
            overlayImg = insertShape(overlayImg, 'Line', lines, 'Color', 'cyan', 'LineWidth', 2);
        catch
            rRange = max(1, cy-8):min(h, cy+8);
            cRange = max(1, cx-8):min(w, cx+8);
            overlayImg(cy, cRange, 1:2) = 0; overlayImg(cy, cRange, 3) = 255;
            overlayImg(rRange, cx, 1:2) = 0; overlayImg(rRange, cx, 3) = 255;
        end
    end
end
