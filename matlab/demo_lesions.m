function summary = demo_lesions(imagePath, onnxPath, thresholdsPath, outPng)
%DEMO_LESIONS Opt-in learned lesion candidates (EX, HE, MA, SE) for one fundus image; writes an overlay PNG.
%   Research candidates, not a diagnosis (locked DDR test Dice: EX 0.60, HE 0.43, MA 0.26). Does not touch main_pipeline defaults.
    arguments
        imagePath (1,1) string
        onnxPath (1,1) string
        thresholdsPath (1,1) string
        outPng (1,1) string = "lesion_candidates.png"
    end
    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, 'src', 'm2_lesion'));
    rgb = imread(imagePath);
    if size(rgb, 3) == 1, rgb = repmat(rgb, 1, 1, 3); end
    summary = runLearnedLesions(rgb, onnxPath, thresholdsPath, outPng);
    for k = 1:4
        fprintf('%s: %d components, %.2f%% of the field\n', summary.names{k}, summary.count(k), 100 * summary.areaFraction(k));
    end
end
