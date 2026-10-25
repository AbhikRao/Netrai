function summary = summariseLearnedLesions(probs, field, thresholds)
%SUMMARISELEARNEDLESIONS Threshold learned lesion probabilities and count candidates inside the field of view.
%   summary = summariseLearnedLesions(probs, field, thresholds)
%   probs       H-by-W-by-4 probabilities (EX, HE, MA, SE)
%   field       H-by-W logical field-of-view mask
%   thresholds  1-by-4 [EX HE MA SE] (validation-chosen; MA uses the lesion-level operating point)
%   summary.masks (H-by-W-by-4 logical), summary.count (1-by-4 connected components), summary.areaFraction (1-by-4)
    arguments
        probs (:,:,4) {mustBeNumeric}
        field (:,:) logical
        thresholds (1,4) double
    end
    assert(isequal(size(field), [size(probs, 1) size(probs, 2)]), 'summariseLearnedLesions:size', 'field and probs differ in size');
    masks = false(size(probs));
    count = zeros(1, 4);
    areaFraction = zeros(1, 4);
    fieldPixels = max(nnz(field), 1);
    for k = 1:4
        m = probs(:, :, k) >= thresholds(k) & field;
        masks(:, :, k) = m;
        count(k) = bwconncomp(m, 8).NumObjects;
        areaFraction(k) = nnz(m) / fieldPixels;
    end
    summary = struct('names', {{'EX', 'HE', 'MA', 'SE'}}, 'masks', masks, 'count', count, 'areaFraction', areaFraction);
end
