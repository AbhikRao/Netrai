function tests = testLearnedLesionsSummary
%TESTLEARNEDLESIONSSUMMARY Unit tests for the opt-in learned lesion summary (no network needed).
tests = functiontests(localfunctions);
end

function setupOnce(~)
addpath(genpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'src')));
end

function testCountsAndAreaInsideField(testCase)
probs = zeros(60, 60, 4);
probs(5:9, 5:9, 1) = 0.9;          % EX blob 1 (25 px)
probs(20:22, 20:22, 1) = 0.9;      % EX blob 2 (9 px)
probs(30:33, 30:33, 2) = 0.95;     % HE blob
probs(1, 1, 3) = 0.99;             % MA outside the field: ignored
field = false(60); field(3:58, 3:58) = true;
s = summariseLearnedLesions(probs, field, [0.5 0.5 0.5 0.5]);
verifyEqual(testCase, s.count, [2 1 0 0]);
verifyEqual(testCase, s.areaFraction(1), 34 / nnz(field), 'AbsTol', 1e-12);
verifyFalse(testCase, any(s.masks(1, 1, :), 'all'));
end

function testThresholdsArePerLesion(testCase)
probs = repmat(0.6, 10, 10, 4);
s = summariseLearnedLesions(probs, true(10), [0.5 0.7 0.5 0.7]);
verifyEqual(testCase, s.count, [1 0 1 0]);
end

function testSizeMismatchIsAnError(testCase)
verifyError(testCase, @() summariseLearnedLesions(zeros(5, 5, 4), true(4), [.5 .5 .5 .5]), 'summariseLearnedLesions:size');
end
