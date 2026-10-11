function vesselMask = segmentVessels(greenChannel, fovMask)
%SEGMENTVESSELS Segments retinal blood vessels.
%   vesselMask = segmentVessels(greenChannel, fovMask)
%   Inputs:
%       greenChannel - Green channel of the image (uint8 or double)
%       fovMask - Field of view mask
%   Outputs:
%       vesselMask - Binary mask of segmented vessels

    if isempty(greenChannel) || ~(isnumeric(greenChannel) || islogical(greenChannel)) ...
            || ~ismatrix(greenChannel) || ~isreal(greenChannel) ...
            || any(~isfinite(greenChannel), 'all')
        error('NetrAI:InvalidVesselInput', ...
            'Green channel must be a nonempty, finite, real 2-D image.');
    end

    if nargin > 1 && ~isempty(fovMask)
        if ~(isnumeric(fovMask) || islogical(fovMask)) || ~isreal(fovMask) ...
                || ~isequal(size(fovMask), size(greenChannel)) ...
                || any(~isfinite(fovMask), 'all')
            error('NetrAI:InvalidVesselMask', ...
                'Field-of-view mask must be finite and match the green image size.');
        end
        validFov = logical(fovMask);
    else
        validFov = true(size(greenChannel));
    end
    vesselMask = false(size(greenChannel));
    if ~any(validFov, 'all')
        return;
    end
    
    % Ensure greenChannel is double in [0, 1]
    greenDbl = im2double(greenChannel);
    
    % CLAHE enhancement
    enhanced = adapthisteq(greenDbl, 'NumTiles', [8 8], 'ClipLimit', 0.01);
    
    % Multi-scale, multi-orientation line openings.  This replaces the old
    % Gabor/Otsu implementation after its DRIVE baseline measured only 0.13
    % sensitivity and 0.17 Dice.
    maxResponse = zeros(size(greenChannel), 'double');
    angles = 0:15:165;
    invEnhanced = imcomplement(enhanced);
    scale = min(size(greenChannel, 1), size(greenChannel, 2)) / 512;
    lengths = unique(max(3, 2 * floor(([7 11 15] * scale) / 2) + 1));
    for lineLength = lengths
        for ang = angles
            response = imopen(invEnhanced, strel('line', lineLength, ang));
            maxResponse = max(maxResponse, double(response));
        end
    end

    validResponse = maxResponse(validFov);
    if any(~isfinite(validResponse))
        error('NetrAI:InvalidVesselInput', 'Vessel response must be finite.');
    end
    if max(validResponse) == min(validResponse)
        % A percentile of a constant response marks the entire FOV with >=.
        % Without contrast there is no supported vessel candidate.
        return;
    end
    threshold = prctile(validResponse, 86);
    bw = maxResponse >= threshold;
    bw = imopen(bw, ones(2));
    bw = bwareaopen(bw, max(8, round(0.00003 * numel(bw))));
    
    vesselMask = bw & validFov;
end
