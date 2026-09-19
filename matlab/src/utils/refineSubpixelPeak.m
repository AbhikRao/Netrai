function [xRefined, yRefined] = refineSubpixelPeak(response, x, y)
%REFINESUBPIXELPEAK Quadratic refinement around an integer local maximum.

    [h, w] = size(response);
    x = round(x); y = round(y);
    if x <= 1 || x >= w || y <= 1 || y >= h
        xRefined = double(x); yRefined = double(y); return;
    end
    center = double(response(y,x));
    denominatorX = double(response(y,x-1)) - 2*center + double(response(y,x+1));
    denominatorY = double(response(y-1,x)) - 2*center + double(response(y+1,x));
    offsetX = 0; offsetY = 0;
    if denominatorX < -1e-12
        offsetX = 0.5 * (double(response(y,x-1))-double(response(y,x+1))) / denominatorX;
    end
    if denominatorY < -1e-12
        offsetY = 0.5 * (double(response(y-1,x))-double(response(y+1,x))) / denominatorY;
    end
    xRefined = double(x) + min(max(offsetX,-0.5),0.5);
    yRefined = double(y) + min(max(offsetY,-0.5),0.5);
end
