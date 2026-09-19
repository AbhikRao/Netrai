function [iqs, metrics] = assessImageQuality(rawImg)
%ASSESSIMAGEQUALITY MATLAB equivalent of Python M1 quality scoring.

    if nargin < 1 || isempty(rawImg)
        error('NetrAI:EmptyImage', 'Input RGB image is required.');
    end
    if ndims(rawImg) ~= 3 || size(rawImg,3) ~= 3
        error('NetrAI:InvalidImage', 'Input must be a 3-channel RGB image.');
    end
    rawImg = im2uint8(rawImg);
    green = double(rawImg(:,:,2));
    [rows, cols] = size(green);

    laplacianKernel = [0 1 0; 1 -4 1; 0 1 0];
    laplacian = imfilter(green, laplacianKernel, 'replicate', 'conv');
    focus = min(var(laplacian(:)) / max(200, 0.002*rows*cols), 1);

    r = floor(rows/2); c = floor(cols/2);
    means = [mean(green(1:r,1:c),'all'), mean(green(1:r,c+1:end),'all'), ...
        mean(green(r+1:end,1:c),'all'), mean(green(r+1:end,c+1:end),'all')];
    illumination = min(means) / (max(means) + 1e-5);
    fov = nnz(rgb2gray(rawImg) > 10) / (rows*cols);
    iqs = 0.40*focus + 0.35*illumination + 0.25*fov;
    metrics = struct('focus',focus,'illumination',illumination,'fov',fov,'iqs',iqs);
end
