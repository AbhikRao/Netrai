function summary = demo_lesions(imagePath, onnxPath, thresholdsPath, outPng)
%DEMO_LESIONS Opt-in learned lesion candidates (EX, HE, MA, SE) for one fundus image.
%   summary = demo_lesions(imagePath, onnxPath, thresholdsPath, outPng)
%   Crops to the field of view, scales the long side to at most 1536 pixels (area-style antialiased resize; OpenCV INTER_AREA
%   is not bit-identical), runs segmentLesionsLearned, thresholds with the validation-chosen thresholds (thresholds.json:
%   dice_thresholds, ma_lesion_threshold) and writes an overlay PNG. Research candidates, not a diagnosis: pixel Dice on the
%   locked DDR test was EX 0.60, HE 0.43, MA 0.26. Does not touch main_pipeline.
    arguments
        imagePath (1,1) string
        onnxPath (1,1) string
        thresholdsPath (1,1) string
        outPng (1,1) string = "lesion_candidates.png"
    end
    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, 'src', 'm2_lesion'));
    cfg = jsondecode(fileread(thresholdsPath));
    thr = [cfg.dice_thresholds.EX, cfg.dice_thresholds.HE, cfg.ma_lesion_threshold, cfg.dice_thresholds.SE];
    rgb = imread(imagePath);
    if size(rgb, 3) == 1, rgb = repmat(rgb, 1, 1, 3); end
    fg = max(rgb, [], 3) > 7;
    rows = find(any(fg, 2)); cols = find(any(fg, 1));
    crop = rgb(rows(1):rows(end), cols(1):cols(end), :);
    scale = min(1, 1536 / max(size(crop, 1), size(crop, 2)));
    if scale < 1
        crop = imresize(crop, scale, 'bilinear', 'Antialiasing', true);
    end
    field = max(crop, [], 3) > 7;
    out = segmentLesionsLearned(crop, 'OnnxPath', onnxPath);
    names = out.names; colors = uint8([255 205 0; 230 30 60; 0 190 255; 255 255 255]);
    overlay = crop;
    counts = zeros(1, 4); areaFraction = zeros(1, 4);
    for k = 1:4
        mask = out.probs(:, :, k) >= thr(k) & field;
        counts(k) = bwconncomp(mask, 8).NumObjects;
        areaFraction(k) = nnz(mask) / max(nnz(field), 1);
        for c = 1:3
            channel = overlay(:, :, c);
            channel(mask) = uint8(0.45 * double(channel(mask)) + 0.55 * double(colors(k, c)));
            overlay(:, :, c) = channel;
        end
    end
    imwrite(overlay, outPng);
    summary = struct('names', {names}, 'count', counts, 'areaFraction', areaFraction, 'png', outPng);
    for k = 1:4
        fprintf('%s: %d components, %.2f%% of the field\n', names{k}, counts(k), 100 * areaFraction(k));
    end
end
