function tests = testAssessGradCAMQuality
    tests = functiontests(localfunctions);
end

function testMalformedAttributionCannotBecomeEvidence(testCase)
    invalid = {[], ones(2,2,2), [1 NaN; 0 0], [1 Inf; 0 0], ...
        [1 -1; 0 0], complex(ones(4), ones(4))};
    for index = 1:numel(invalid)
        value = invalid{index};
        verifyError(testCase, @() assessGradCAMQuality(value, true(4)), ...
            'NetrAI:InvalidGradCAM');
    end
end

function testMalformedFovCannotSilentlyCountAsRetinalSupport(testCase)
    invalid = {[], ones(2,2,2), [1 NaN; 0 0], [1 Inf; 0 0], ...
        [1 0.5; 0 0], [1 -1; 0 0], complex(ones(4), ones(4))};
    for index = 1:numel(invalid)
        value = invalid{index};
        verifyError(testCase, @() assessGradCAMQuality(ones(4), value), ...
            'NetrAI:InvalidFOV');
    end
end

function testZeroMapRemainsAnExplicitWarning(testCase)
    metrics = assessGradCAMQuality(zeros(32), true(32));
    verifyTrue(testCase, metrics.shortcut_flag);
    verifyFalse(testCase, metrics.informative);
    verifyEqual(testCase, metrics.status, "zero_attribution");
    verifyEqual(testCase, metrics.border_attention_fraction, 0);
end

function testZeroMapStillValidatesFov(testCase)
    verifyError(testCase, @() assessGradCAMQuality(zeros(32), NaN(32)), ...
        'NetrAI:InvalidFOV');
end

function testTinyPositiveMapsHaveFiniteUninformativeMetrics(testCase)
    shapes = {[1 1], [2 2], [1 10]};
    for index = 1:numel(shapes)
        shape = shapes{index};
        metrics = assessGradCAMQuality(ones(shape), true(shape));
        verifyTrue(testCase, metrics.shortcut_flag);
        verifyFalse(testCase, metrics.informative);
        verifyEqual(testCase, metrics.status, "insufficient_spatial_grid");
        names = fieldnames(metrics);
        for fieldIndex = 1:numel(names)
            value = metrics.(names{fieldIndex});
            if isnumeric(value)
                verifyTrue(testCase, isfinite(value));
            end
        end
    end
end

function testDifferentValidFovResolutionIsSupported(testCase)
    metrics = assessGradCAMQuality(ones(100), true(20));
    verifyTrue(testCase, metrics.informative);
    verifyFalse(testCase, metrics.shortcut_flag);
    verifyEqual(testCase, metrics.outside_fov_attention_fraction, 0);
    verifyEqual(testCase, metrics.border_enrichment, 1);
end

function testOutsideRetinalAttentionRemainsFlagged(testCase)
    heatmap = zeros(40); heatmap(1:10,:) = 1;
    fov = false(20); fov(8:20,:) = true;
    metrics = assessGradCAMQuality(heatmap, fov);
    verifyTrue(testCase, metrics.shortcut_flag);
    verifyTrue(testCase, metrics.informative);
    verifyEqual(testCase, metrics.outside_fov_attention_fraction, 1);
end

function testAttentionSumOverflowIsRejected(testCase)
    verifyError(testCase, @() assessGradCAMQuality(realmax*ones(4), true(4)), ...
        'NetrAI:InvalidGradCAM');
end
