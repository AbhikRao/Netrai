function imageRGB = readScreeningImage(imagePath)
%READSCREENINGIMAGE Decode supported image files into uint8 RGB.
% Indexed images use their color map. Palette indices are not intensities.
% RGB transparency is ignored consistently with the released color input.
% High-bit-depth intensity images require a separately validated contract.
    arguments
        imagePath (1,1) string
    end
    % Request the alpha output so PNG colors are not composited on a background.
    [pixels, colorMap, ~] = imread(imagePath);
    if isempty(pixels)
        error('NetrAI:InvalidImage','Expected a nonempty image.');
    end
    if ~isempty(colorMap)
        if ~ismatrix(pixels) || ~isreal(colorMap) || size(colorMap,2)~=3 || ...
                any(~isfinite(colorMap(:)) | colorMap(:)<0 | colorMap(:)>1)
            error('NetrAI:InvalidImage','Invalid indexed-image color map.');
        end
        imageRGB = im2uint8(ind2rgb(pixels,colorMap));
        return;
    end
    if islogical(pixels)
        pixels = uint8(pixels)*255;
    end
    if ~isa(pixels,'uint8')
        error('NetrAI:InvalidImage', ...
            'Supported intensity input is uint8; high-bit-depth images are not silently rescaled.');
    end
    if ismatrix(pixels)
        imageRGB = repmat(pixels,1,1,3);
    elseif ndims(pixels)==3 && size(pixels,3)==3
        imageRGB = pixels;
    else
        error('NetrAI:InvalidImage','Expected grayscale, indexed-color or RGB input.');
    end
end
