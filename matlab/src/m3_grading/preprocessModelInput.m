function [modelInput, modelFov] = preprocessModelInput(imageRGB, config)
%PREPROCESSMODELINPUT Reproduce the Python training preprocessing in MATLAB.
% Output is single H-by-W-by-3 in ImageNet-normalized RGB order.

    if isempty(imageRGB)
        error('NetrAI:EmptyImage', 'Input image cannot be empty.');
    end
    if ismatrix(imageRGB)
        imageRGB = repmat(imageRGB, 1, 1, 3);
    end
    if ndims(imageRGB) ~= 3 || size(imageRGB, 3) ~= 3
        error('NetrAI:InvalidImage', 'Input must be an H-by-W-by-3 RGB image.');
    end
    imageRGB = im2uint8(imageRGB);

    threshold = 7;
    if isfield(config, 'black_crop_threshold')
        threshold = double(config.black_crop_threshold);
    end
    foreground = max(imageRGB, [], 3) > threshold;
    if any(foreground(:))
        rows = find(any(foreground, 2));
        cols = find(any(foreground, 1));
        imageRGB = imageRGB(rows(1):rows(end), cols(1):cols(end), :);
        foreground = foreground(rows(1):rows(end), cols(1):cols(end));
    end

    targetSize = double(config.input_size);
    % A measured cross-runtime audit found bicubic resampling plus uint8
    % Gaussian/blend stages substantially closer to the deployed OpenCV path
    % than the former box/float approximation. Keep uint8 rounding at the
    % same two stages where OpenCV returns uint8 outputs.
    resized = imresize(imageRGB, [targetSize targetSize], 'bicubic');
    sigma = double(config.ben_graham_sigma);
    blurred = imgaussfilt(resized, sigma, ...
        'FilterSize', 2*ceil(3*sigma)+1, 'Padding','replicate');
    enhanced = imlincomb(4, resized, -4, blurred, 128, 'uint8');
    enhanced = single(enhanced) ./ 255;

    means = reshape(single(config.imagenet_mean), 1, 1, 3);
    stds = reshape(single(config.imagenet_std), 1, 1, 3);
    modelInput = (enhanced - means) ./ stds;
    modelFov = imresize(foreground, [targetSize targetSize], 'nearest') > 0;
end
