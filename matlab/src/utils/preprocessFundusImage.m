function [procImg, greenCh, mask] = preprocessFundusImage(rawImg)
%PREPROCESSFUNDUSIMAGE Preprocesses a raw fundus image.
%
%   [procImg, greenCh, mask] = preprocessFundusImage(rawImg)
%
%   Inputs:
%       rawImg - Raw RGB fundus image
%
%   Outputs:
%       procImg - Resized (512x512) and padded RGB image
%       greenCh - Extracted green channel
%       mask    - Binary circular FOV mask

    % Check input
    if isempty(rawImg)
        error('Input image is empty.');
    end
    if size(rawImg, 3) ~= 3
        warning('Input image does not have 3 channels. Assuming grayscale.');
        rawImg = cat(3, rawImg, rawImg, rawImg);
    end

    targetSize = [512, 512];
    [h, w, ~] = size(rawImg);
    
    % Resize preserving aspect ratio and pad
    scale = min(targetSize(1)/h, targetSize(2)/w);
    newH = round(h * scale);
    newW = round(w * scale);
    
    resizedImg = imresize(rawImg, [newH, newW]);
    
    % Pad to target size
    padH = targetSize(1) - newH;
    padW = targetSize(2) - newW;
    
    padTop = floor(padH / 2);
    padBottom = ceil(padH / 2);
    padLeft = floor(padW / 2);
    padRight = ceil(padW / 2);
    
    procImg = padarray(resizedImg, [padTop, padLeft], 0, 'pre');
    procImg = padarray(procImg, [padBottom, padRight], 0, 'post');
    
    % Extract green channel
    greenCh = procImg(:, :, 2);
    
    % Create FOV mask via thresholding
    % Assuming background is very dark (pixels > 15 as per standard)
    grayImg = rgb2gray(procImg);
    mask = imbinarize(grayImg, 15/255);
    % Fill holes and clean up
    mask = imfill(mask, 'holes');
    mask = bwareaopen(mask, 1000); % remove small artifacts
end
