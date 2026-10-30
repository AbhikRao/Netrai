function summary = summarizeSiteVerification(called, referable, targetSensitivity, targetSpecificity)
%SUMMARIZESITEVERIFICATION Sensitivity/specificity of a site policy on a separate labelled sample, with Wilson 95% intervals.
% Same definitions as verification_summary in python/calibrate_site.py.
    arguments
        called (:,1) logical
        referable (:,1) logical
        targetSensitivity (1,1) double = 0.90
        targetSpecificity (1,1) double = 0.85
    end
    if numel(called) ~= numel(referable)
        error('NetrAI:SiteSampleMismatch', 'Decisions and labels differ in length.');
    end
    tp = nnz(called & referable); fn = nnz(~called & referable);
    tn = nnz(~called & ~referable); fp = nnz(called & ~referable);
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1);
    if tp + fn == 0, sens = NaN; end
    if tn + fp == 0, spec = NaN; end
    sensCI = wilsonInterval(tp, tp + fn); specCI = wilsonInterval(tn, tn + fp);
    summary = struct('referable', tp + fn, 'non_referable', tn + fp, 'tp', tp, 'fn', fn, 'tn', tn, 'fp', fp, ...
        'sensitivity', sens, 'sensitivity_ci95', sensCI, 'specificity', spec, 'specificity_ci95', specCI, ...
        'meets_targets_point', sens >= targetSensitivity && spec >= targetSpecificity, ...
        'meets_targets_lower_bound', sensCI(1) >= targetSensitivity && specCI(1) >= targetSpecificity, ...
        'targets', struct('sensitivity', targetSensitivity, 'specificity', targetSpecificity));
end

function ci = wilsonInterval(successes, total)
    if total == 0
        ci = [NaN NaN]; return
    end
    z = 1.959964; p = successes / total;
    centre = (p + z^2 / (2 * total)) / (1 + z^2 / total);
    half = z * sqrt(p * (1 - p) / total + z^2 / (4 * total^2)) / (1 + z^2 / total);
    ci = [centre - half, centre + half];
end
