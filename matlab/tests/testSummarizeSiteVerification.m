function tests = testSummarizeSiteVerification
%TESTSUMMARIZESITEVERIFICATION Unit tests for the site-policy verification summary (no model needed).
tests = functiontests(localfunctions);
end

function setupOnce(~)
addpath(genpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'src')));
end

function testCountsRatesAndWilsonIntervals(testCase)
called = [true(9,1); false(1,1); false(170,1); true(30,1)];
referable = [true(10,1); false(200,1)];
s = summarizeSiteVerification(called, referable);
verifyEqual(testCase, [s.tp s.fn s.tn s.fp], [9 1 170 30]);
verifyEqual(testCase, s.sensitivity, 0.9, 'AbsTol', 1e-12);
verifyEqual(testCase, s.sensitivity_ci95, [0.5958 0.9821], 'AbsTol', 1e-3);   % same Wilson interval as python wilson(9, 10)
verifyEqual(testCase, s.specificity, 0.85, 'AbsTol', 1e-12);
verifyTrue(testCase, s.meets_targets_point);
verifyFalse(testCase, s.meets_targets_lower_bound);
end

function testEmptyClassAndMismatch(testCase)
s = summarizeSiteVerification(false(5,1), false(5,1));
verifyTrue(testCase, isnan(s.sensitivity));
verifyError(testCase, @() summarizeSiteVerification(false(5,1), false(4,1)), 'NetrAI:SiteSampleMismatch');
end
