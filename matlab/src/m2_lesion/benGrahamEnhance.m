function out = benGrahamEnhance(rgb, sigma)
%BENGRAHAMENHANCE  4*I - 4*GaussianBlur(I) + 128 on uint8 (OpenCV addWeighted semantics, REFLECT_101 borders).
    k = bitor(round(sigma * 6 + 1), 1);                          % OpenCV kernel size for uint8 input
    r = (k - 1) / 2;
    xs = -r:r;
    g = exp(-(xs .^ 2) / (2 * sigma ^ 2));
    g = g / sum(g);
    blurred = zeros(size(rgb), 'uint8');
    for c = 1:3
        im = double(rgb(:, :, c));
        [hh, ww] = size(im);
        ri = reflect101(1 - r : hh + r, hh);
        ci = reflect101(1 - r : ww + r, ww);
        padded = im(ri, ci);
        tmp = conv2(padded, g(:), 'valid');                      % vertical pass
        tmp = conv2(tmp, g(:)', 'valid');                        % horizontal pass
        blurred(:, :, c) = uint8(round(tmp));
    end
    out = uint8(min(max(4 * double(rgb) - 4 * double(blurred) + 128, 0), 255));
end

function idx = reflect101(idx, n)
%REFLECT101  Index reflection without repeating the edge (OpenCV BORDER_REFLECT_101).
    if n == 1
        idx = ones(size(idx));
        return
    end
    period = 2 * (n - 1);
    idx = mod(idx - 1, period);
    idx(idx >= n) = period - idx(idx >= n);
    idx = idx + 1;
end
