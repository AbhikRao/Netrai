function modelInput = preprocessQualityModelInput(imageRGB, config)
%PREPROCESSQUALITYMODELINPUT Exact half-pixel bilinear candidate preprocessing.

    if isempty(imageRGB)
        error('NetrAI:EmptyImage', 'Input image cannot be empty.');
    end
    if ismatrix(imageRGB)
        imageRGB = repmat(imageRGB, 1, 1, 3);
    end
    if ndims(imageRGB) ~= 3 || size(imageRGB, 3) ~= 3
        error('NetrAI:InvalidImage', 'Input must be H-by-W-by-3 RGB.');
    end
    imageRGB = im2uint8(imageRGB);
    target = double(config.input_size);
    height = size(imageRGB, 1);
    width = size(imageRGB, 2);
    yy = ((0:target-1) + 0.5) .* height ./ target - 0.5;
    xx = ((0:target-1) + 0.5) .* width ./ target - 0.5;
    yy = min(max(yy, 0), height - 1);
    xx = min(max(xx, 0), width - 1);
    y0 = floor(yy); x0 = floor(xx);
    y1 = min(y0 + 1, height - 1); x1 = min(x0 + 1, width - 1);
    wy = single(yy - y0)';
    wx = single(xx - x0);
    pixels = single(imageRGB);
    top = pixels(y0+1, x0+1, :) .* reshape(1-wx, 1, [], 1) + ...
        pixels(y0+1, x1+1, :) .* reshape(wx, 1, [], 1);
    bottom = pixels(y1+1, x0+1, :) .* reshape(1-wx, 1, [], 1) + ...
        pixels(y1+1, x1+1, :) .* reshape(wx, 1, [], 1);
    resized = top .* reshape(1-wy, [], 1, 1) + ...
        bottom .* reshape(wy, [], 1, 1);
    resized = resized ./ single(255);
    means = reshape(single(config.imagenet_mean), 1, 1, 3);
    stds = reshape(single(config.imagenet_std), 1, 1, 3);
    modelInput = (resized - means) ./ stds;
end
