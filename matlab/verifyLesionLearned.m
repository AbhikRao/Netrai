function report = verifyLesionLearned(fixtureDir, onnxPath, outFile)
%VERIFYLESIONLEARNED Parity of the MATLAB learned-lesion module against the Python/ONNX Runtime reference.
%   Measures (1) preprocessing parity (MATLAB Ben Graham enhancement vs Python for the first window), (2) network parity on
%   identical enhanced input, and (3) end-to-end full-image probability and frozen-threshold mask agreement.
    arguments
        fixtureDir (1,1) string
        onnxPath (1,1) string
        outFile (1,1) string = ""
    end
    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, 'src', 'm2_lesion'));
    cfg = jsondecode(fileread(fullfile(fixtureDir, 'fixtures.json')));
    names = {'EX', 'HE', 'MA', 'SE'};
    thr = [cfg.dice_thresholds.EX cfg.dice_thresholds.HE cfg.dice_thresholds.MA cfg.dice_thresholds.SE];
    net = importNetworkFromONNX(onnxPath, 'InputDataFormats', 'BCSS', 'OutputDataFormats', 'BCSS');
    ids = cfg.ids; if ischar(ids), ids = {ids}; end
    report = struct('image', {}, 'enhancedMaxAbsDiff', {}, 'enhancedFracDiff', {}, 'networkMaxAbsProb', {}, ...
        'endToEndMaxAbsProb', {}, 'endToEndMeanAbsProb', {}, 'maskDisagreeFraction', {});
    for i = 1:numel(ids)
        id = ids{i};
        work = imread(fullfile(fixtureDir, [id '_work.png']));
        ref = load(fullfile(fixtureDir, [id '_ref.mat']));
        % (1)+(2) first window
        [h, w, ~] = size(work); sigma = max(2, max(h, w) / 100); tile = cfg.tile;
        win = work(1:min(tile, h), 1:min(tile, w), :);
        enhanced = localBenGraham(win, sigma);
        refEnh = uint8(ref.enhanced_window0);
        dEnh = abs(double(enhanced) - double(refEnh));
        % network on the PYTHON enhanced window (isolates the network)
        padded = zeros(tile, tile, 3, 'uint8'); padded(1:size(refEnh,1), 1:size(refEnh,2), :) = refEnh;
        x = (single(padded) / 255 - reshape(single([0.485 0.456 0.406]), 1, 1, 3)) ./ reshape(single([0.229 0.224 0.225]), 1, 1, 3);
        y = extractdata(predict(net, dlarray(reshape(permute(x, [3 1 2]), [1 3 tile tile]), 'BCSS')));
        p = 1 ./ (1 + exp(-y(1:size(refEnh,1), 1:size(refEnh,2), :, 1)));
        netDiff = max(abs(single(p) - single(ref.window0_probs)), [], 'all');
        % (3) end to end
        out = segmentLesionsLearned(work, 'Net', net, 'Tile', tile, 'Overlap', cfg.overlap);
        d = abs(out.probs - single(ref.probs));
        dis = 0;
        for k = 1:4
            dis = dis + mean((out.probs(:,:,k) >= thr(k)) ~= (single(ref.probs(:,:,k)) >= thr(k)), 'all');
        end
        report(i).image = id;
        report(i).enhancedMaxAbsDiff = max(dEnh, [], 'all');
        report(i).enhancedFracDiff = mean(dEnh > 0, 'all');
        report(i).networkMaxAbsProb = double(netDiff);
        report(i).endToEndMaxAbsProb = double(max(d, [], 'all'));
        report(i).endToEndMeanAbsProb = double(mean(d, 'all'));
        report(i).maskDisagreeFraction = dis / 4;
        fprintf('%s: enhanced maxdiff %g (frac %.4f), network max %.3g, end-to-end max %.3g mean %.3g, mask disagree %.2e\n', id, ...
            report(i).enhancedMaxAbsDiff, report(i).enhancedFracDiff, report(i).networkMaxAbsProb, report(i).endToEndMaxAbsProb, ...
            report(i).endToEndMeanAbsProb, report(i).maskDisagreeFraction);
    end
    if strlength(outFile) > 0
        fid = fopen(outFile, 'w'); fwrite(fid, jsonencode(report, 'PrettyPrint', true)); fclose(fid);
    end
end

function out = localBenGraham(rgb, sigma)
    out = benGrahamEnhance(rgb, sigma);
end
