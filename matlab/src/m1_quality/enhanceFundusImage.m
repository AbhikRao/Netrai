function enhancedImg = enhanceFundusImage(rawImg)
%ENHANCEFUNDUSIMAGE Adaptive enhancement pipeline for borderline fundus images.
% enhancedImg = enhanceFundusImage(rawImg)
% Inputs: rawImg - RGB uint8 fundus image
% Outputs: enhancedImg - enhanced RGB uint8 fundus image

    % Input validation
    if nargin < 1 || isempty(rawImg)
        error('Input rawImg is required and cannot be empty.');
    end
    if ndims(rawImg) ~= 3 || size(rawImg, 3) ~= 3
        error('Input must be a 3-channel RGB image.');
    end
    if ~isa(rawImg, 'uint8')
        rawImg = im2uint8(rawImg);
    end

    % 1. Background illumination subtraction on green channel
    green = rawImg(:,:,2);
    bg = imgaussfilt(green, 45);
    green_sub = imsubtract(green, bg);
    green_adj = imadjust(green_sub);
    
    tempImg = rawImg;
    tempImg(:,:,2) = green_adj;

    % 2. Convert to LAB, apply adapthisteq on L channel
    labImg = rgb2lab(tempImg);
    L = labImg(:,:,1) / 100; % Normalize to [0, 1] for adapthisteq
    L_enh = adapthisteq(L, 'ClipLimit', 0.02, 'NumTiles', [8 8], 'Distribution', 'rayleigh');
    labImg(:,:,1) = L_enh * 100;

    % 3. Convert back to RGB
    rgbImg = lab2rgb(labImg);

    % 4. Apply imguidedfilter on each channel (DegreeOfSmoothing 0.01)
    filteredImg = zeros(size(rgbImg));
    for c = 1:3
        filteredImg(:,:,c) = imguidedfilter(rgbImg(:,:,c), 'DegreeOfSmoothing', 0.01);
    end

    % 5. Return im2uint8 result
    enhancedImg = im2uint8(filteredImg);
end
