function result = segmentLesionsLearned(workRgb, options)
%SEGMENTLESIONSLEARNED Learned lesion probabilities (EX, HE, MA, SE) by sliding windows (opt-in M2 module).
%   result = segmentLesionsLearned(workRgb, 'OnnxPath', p) runs the exported lesion segmenter
%   (U-Net, EfficientNet-B2 encoder, fixed 1x3x1024x1024 ONNX input) over a work image: uint8 RGB already cropped to the
%   field of view and scaled so that its long side is at most 1536 pixels (the training scale). The windowing mirrors the
%   Python reference (python/train_lesion_seg.py predict_full): 1024-pixel windows, 128-pixel overlap, Ben Graham
%   enhancement per window with sigma = max(2, longSide/100), probabilities averaged where windows overlap. Windows are
%   zero-padded after enhancement to the fixed ONNX input size.
%
%   result.probs   H-by-W-by-4 single, sigmoid probabilities in the order EX, HE, MA, SE
%   result.names   {'EX','HE','MA','SE'}
%   Optional 'Net' reuses an imported dlnetwork. This module is opt-in: it does not change main_pipeline defaults.
    arguments
        workRgb (:,:,3) uint8
        options.OnnxPath (1,1) string = ""
        options.Net = []
        options.Tile (1,1) double = 1024
        options.Overlap (1,1) double = 128
    end
    net = options.Net;
    if isempty(net)
        assert(strlength(options.OnnxPath) > 0 && isfile(options.OnnxPath), 'segmentLesionsLearned:onnx', 'ONNX file not found');
        net = importNetworkFromONNX(options.OnnxPath, 'InputDataFormats', 'BCSS', 'OutputDataFormats', 'BCSS');
    end
    tile = options.Tile; overlap = options.Overlap;
    [h, w, ~] = size(workRgb);
    sigma = max(2, max(h, w) / 100);
    step = max(32, tile - overlap);
    tops = 0:step:(max(1, h - overlap) - 1);
    lefts = 0:step:(max(1, w - overlap) - 1);
    probs = zeros(h, w, 4, 'single');
    counts = zeros(h, w, 'single');
    for top = tops
        for left = lefts
            rows = top + 1 : min(top + tile, h);
            cols = left + 1 : min(left + tile, w);
            enhanced = benGrahamEnhance(workRgb(rows, cols, :), sigma);
            padded = zeros(tile, tile, 3, 'uint8');
            padded(1:numel(rows), 1:numel(cols), :) = enhanced;
            x = (single(padded) / 255 - reshape(single([0.485 0.456 0.406]), 1, 1, 3)) ./ reshape(single([0.229 0.224 0.225]), 1, 1, 3);
            x = reshape(permute(x, [3 1 2]), [1 3 tile tile]);
            y = predict(net, dlarray(x, 'BCSS'));
            y = extractdata(y);                                  % H-by-W-by-4 (labels SSCB)
            p = 1 ./ (1 + exp(-y(1:numel(rows), 1:numel(cols), :, 1)));
            probs(rows, cols, :) = probs(rows, cols, :) + single(p);
            counts(rows, cols) = counts(rows, cols) + 1;
        end
    end
    result.probs = probs ./ max(counts, 1);
    result.names = {'EX', 'HE', 'MA', 'SE'};
end
