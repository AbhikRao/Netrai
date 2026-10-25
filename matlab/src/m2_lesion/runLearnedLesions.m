function result = runLearnedLesions(rgb, onnxPath, thresholdsPath, outPng)
%RUNLEARNEDLESIONS Opt-in learned lesion candidates for one RGB image (crop, scale to <= 1536, segment, summarise, overlay).
%   Research module: pixel Dice on the locked DDR test was EX 0.60, HE 0.43, MA 0.26. Candidate evidence, not a diagnosis.
    arguments
        rgb (:,:,3) uint8
        onnxPath (1,1) string
        thresholdsPath (1,1) string
        outPng (1,1) string = ""
    end
    cfg = jsondecode(fileread(thresholdsPath));
    thr = [cfg.dice_thresholds.EX, cfg.dice_thresholds.HE, cfg.ma_lesion_threshold, cfg.dice_thresholds.SE];
    fg = max(rgb, [], 3) > 7;
    rows = find(any(fg, 2)); cols = find(any(fg, 1));
    crop = rgb(rows(1):rows(end), cols(1):cols(end), :);
    scale = min(1, 1536 / max(size(crop, 1), size(crop, 2)));
    if scale < 1
        crop = imresize(crop, scale, 'bilinear', 'Antialiasing', true);
    end
    out = segmentLesionsLearned(crop, 'OnnxPath', onnxPath);
    summary = summariseLearnedLesions(out.probs, max(crop, [], 3) > 7, thr);
    overlay = crop;
    colors = uint8([255 205 0; 230 30 60; 0 190 255; 255 255 255]);
    for k = 1:4
        for c = 1:3
            channel = overlay(:, :, c);
            m = summary.masks(:, :, k);
            channel(m) = uint8(0.45 * double(channel(m)) + 0.55 * double(colors(k, c)));
            overlay(:, :, c) = channel;
        end
    end
    if strlength(outPng) > 0
        imwrite(overlay, outPng);
    end
    result = struct('status', "research opt-in", 'names', {summary.names}, 'count', summary.count, ...
        'areaFraction', summary.areaFraction, 'thresholds', thr, 'overlayPath', outPng);
end
