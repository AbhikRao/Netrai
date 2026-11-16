function tests = testFitSitePolicy
%TESTFITSITEPOLICY Unit tests for the MATLAB site-calibration rule (no model needed).
tests = functiontests(localfunctions);
end

function setupOnce(~)
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
addpath(genpath(fullfile(root, 'src')));
end

function config = baseConfig()
config = struct('grade_thresholds', [1.0687 1.0923 2.0296 2.7971], 'referable_head_threshold', 0.18, ...
    'deployment_identity', struct('model_sha256', repmat('a', 1, 64), 'config_sha256', repmat('b', 1, 64)));
end

function [scores, grades] = sample(seed, nPos, nNeg)
rng(seed);
scores = [1.6 + 0.5 * randn(nPos, 1); 0.5 + 0.4 * randn(nNeg, 1)];
scores = min(max(scores, 0), 4);
grades = [2 * ones(nPos, 1); zeros(nNeg, 1)];
end

function testCutReachesTargetAndThresholdsStayStrictlyIncreasing(testCase)
[scores, grades] = sample(1, 80, 220);
policy = fitSitePolicy(scores, grades, baseConfig());
t = policy.thresholds;
verifyTrue(testCase, all(diff(t) > 0));
verifyEqual(testCase, t(2), policy.site.fitted_boundary);
sens = mean(scores(grades >= 2) >= t(2));
verifyGreaterThanOrEqual(testCase, sens, 0.95);
verifyLessThan(testCase, sens, 0.97);                                   % highest cut reaching the target
verifyEqual(testCase, policy.model_sha256, repmat('a', 1, 64));
verifyEqual(testCase, policy.site.target_sample_sensitivity, 0.95);
end

function testSameRuleAsPython(testCase)
scores = [0.2 0.3 0.4 0.5 0.6 1.1 1.3 1.5 1.7 1.9 2.1 2.3 2.5 2.7 2.9]';
grades = [0 0 0 0 0 2 2 2 2 2 2 2 2 2 2]';
policy = fitSitePolicy(scores, grades, baseConfig(), TargetSensitivity=0.92);
% 10 referable, floor(0.08 * 10) = 0 misses allowed -> cut just below the lowest referable score (1.1), as python fit_boundary
verifyEqual(testCase, policy.thresholds(2), 1.1 - 1e-9, 'AbsTol', 1e-12);
end

function testLowCutKeepsBoundariesValidForGrading(testCase)
scores = [0.05 0.06 0.07 0.08 0.09 0.10 0.11 0.12 0.13 0.14 0.15 0.16]';
grades = [0 0 0 0 0 2 2 2 2 2 2 2]';
policy = fitSitePolicy(scores, grades, baseConfig());
t = policy.thresholds;
verifyLessThan(testCase, t(2), 1.0687);                                % cut below the frozen grade-1 boundary
verifyTrue(testCase, all(diff(t) > 0));                                % predictDRGrade requires strictly increasing boundaries
end

function testPolicyRoundTripsThroughApplySitePolicy(testCase)
[scores, grades] = sample(2, 60, 200);
policy = fitSitePolicy(scores, grades, baseConfig(), SiteName="clinic B");
path = string(fullfile(tempname, 'site.json')); mkdir(fileparts(path));
testCase.addTeardown(@() rmdir(fileparts(path), 's'));
fid = fopen(path, 'w'); fwrite(fid, jsonencode(policy)); fclose(fid);
[config, status] = applySitePolicy(baseConfig(), path);
verifyEqual(testCase, config.grade_thresholds, policy.thresholds, 'AbsTol', 1e-12);
verifyEqual(testCase, status.site_name, "clinic B");
other = baseConfig(); other.deployment_identity.model_sha256 = repmat('c', 1, 64);
verifyError(testCase, @() applySitePolicy(other, path), 'NetrAI:SitePolicyModelMismatch');
end

function testGuardsAndWarnings(testCase)
verifyError(testCase, @() fitSitePolicy([0.1; 0.2; 2], [0; 0; 2], baseConfig()), 'NetrAI:SiteSampleTooSmall');
verifyError(testCase, @() fitSitePolicy([0.1; 0.2], [0; 0; 2], baseConfig()), 'NetrAI:SiteSampleMismatch');
[scores, grades] = sample(3, 10, 100);
policy = fitSitePolicy(scores, grades, baseConfig());
verifyEqual(testCase, numel(policy.site.warnings), 2);
noId = rmfield(baseConfig(), 'deployment_identity');
verifyError(testCase, @() fitSitePolicy(scores, grades, noId), 'NetrAI:SitePolicyModelMismatch');
end

function testCalibrateSiteChecksTheSampleBeforeScoring(testCase)
out = string(tempname) + ".json";
verifyError(testCase, @() calibrateSite(["a.jpg"; "b.jpg"], [0; 2; 2], out, Verbose=false), 'NetrAI:SiteSampleMismatch');
verifyError(testCase, @() calibrateSite(["a.jpg"; "b.jpg"], [0; 2.5], out, Verbose=false), 'MATLAB:validators:mustBeInteger');
verifyError(testCase, @() calibrateSite(["a.jpg"; "b.jpg"], [0; 7], out, Verbose=false), 'MATLAB:validators:mustBeInRange');
verifyFalse(testCase, isfile(out));
end
