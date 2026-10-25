function tests = testApplySitePolicy
%TESTAPPLYSITEPOLICY Unit tests for the per-site referral boundary override (no model needed).
tests = functiontests(localfunctions);
end

function setupOnce(~)
addpath(genpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'src')));
end

function config = baseConfig()
config = struct('grade_thresholds', [1.0 1.2 2.0 2.8], 'referable_head_threshold', 0.2, ...
    'deployment_identity', struct('model_sha256', repmat('a', 1, 64), 'config_sha256', repmat('b', 1, 64)));
end

function path = writePolicy(testCase, policy)
path = string(fullfile(tempname, 'site.json'));
mkdir(fileparts(path));
fid = fopen(path, 'w'); fwrite(fid, jsonencode(policy)); fclose(fid);
testCase.addTeardown(@() rmdir(fileparts(path), 's'));
end

function testNoPolicyKeepsFrozenThresholdsAndFlagsUncalibrated(testCase)
[config, status] = applySitePolicy(baseConfig(), "");
verifyEqual(testCase, config.grade_thresholds, [1.0 1.2 2.0 2.8]);
verifyTrue(testCase, startsWith(status.calibration_status, "UNCALIBRATED"));
end

function testPolicyReplacesThresholdsAndRecordsProvenance(testCase)
policy = struct('status', 'site_calibrated', 'thresholds', [1.0 1.5 2.0 2.8], 'head_threshold', 0.3, ...
    'model_sha256', repmat('a', 1, 64), 'config_sha256', repmat('b', 1, 64), 'site', struct('name', 'clinic A', 'referable', 55));
[config, status] = applySitePolicy(baseConfig(), writePolicy(testCase, policy));
verifyEqual(testCase, config.grade_thresholds, [1.0 1.5 2.0 2.8], 'AbsTol', 1e-12);
verifyEqual(testCase, config.referable_head_threshold, 0.3);
verifyTrue(testCase, contains(status.calibration_status, "site-calibrated"));
verifyEqual(testCase, status.site_name, "clinic A");
verifyEqual(testCase, status.site_referable_cases, 55);
verifyEqual(testCase, strlength(status.site_policy_sha256), 64);
end

function testRejectsMalformedPolicies(testCase)
id = struct('model_sha256', repmat('a', 1, 64), 'config_sha256', repmat('b', 1, 64));
bad = {struct('status', 'site_calibrated', 'thresholds', [1 2 3]), ...
       struct('status', 'site_calibrated', 'thresholds', [1 3 2 4]), ...
       struct('status', 'frozen', 'thresholds', [1 1.5 2 2.8]), ...
       catstruct(struct('status', 'site_calibrated', 'thresholds', [1 1.5 2 2.8], 'head_threshold', 1.5), id)};
for i = 1:numel(bad)
    verifyError(testCase, @() applySitePolicy(baseConfig(), writePolicy(testCase, bad{i})), 'NetrAI:InvalidSitePolicy');
end
verifyError(testCase, @() applySitePolicy(baseConfig(), "/nonexistent/policy.json"), 'NetrAI:SitePolicyNotFound');
end

function s = catstruct(a, b)
s = a;
for f = fieldnames(b)', s.(f{1}) = b.(f{1}); end
end

function testPolicyOfAnotherModelOrWithoutIdentityIsRefused(testCase)
base = struct('status', 'site_calibrated', 'thresholds', [1.0 1.5 2.0 2.8]);
other = catstruct(base, struct('model_sha256', repmat('c', 1, 64), 'config_sha256', repmat('b', 1, 64)));
otherConfig = catstruct(base, struct('model_sha256', repmat('a', 1, 64), 'config_sha256', repmat('d', 1, 64)));
verifyError(testCase, @() applySitePolicy(baseConfig(), writePolicy(testCase, other)), 'NetrAI:SitePolicyModelMismatch');
verifyError(testCase, @() applySitePolicy(baseConfig(), writePolicy(testCase, otherConfig)), 'NetrAI:SitePolicyModelMismatch');
verifyError(testCase, @() applySitePolicy(baseConfig(), writePolicy(testCase, base)), 'NetrAI:SitePolicyModelMismatch');
noIdentity = rmfield(baseConfig(), 'deployment_identity');
okPolicy = catstruct(base, struct('model_sha256', repmat('a', 1, 64), 'config_sha256', repmat('b', 1, 64)));
verifyError(testCase, @() applySitePolicy(noIdentity, writePolicy(testCase, okPolicy)), 'NetrAI:SitePolicyModelMismatch');
end
