function mask = createCircularMask(imgSize, center, radius)
%CREATECIRCULARMASK Create a binary circular mask.
%   mask = createCircularMask(imgSize, center, radius)
%
%   Inputs:
%       imgSize - [rows, cols] of the output mask
%       center  - [x, y] center of the circle
%       radius  - radius in pixels
%   Output:
%       mask - binary mask (logical)

    if nargin < 3
        error('Requires imgSize, center, and radius.');
    end

    [cols, rows] = meshgrid(1:imgSize(2), 1:imgSize(1));
    mask = ((rows - center(2)).^2 + (cols - center(1)).^2) <= radius^2;
end
