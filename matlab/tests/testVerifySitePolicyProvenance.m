function tests = testVerifySitePolicyProvenance
%TESTVERIFYSITEPOLICYPROVENANCE Calibration/verification overlap is refused across runtimes (no model weights needed).
% The fixture policy was written by python/calibrate_site.py fit (python/tests/test_calibrate_site.py checks that fit still writes it):
% its calibration sample contains fixtures/site_policy/cal_1.png and cal_2.png and patients P01-P12; new_1.png is not in it.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
addpath(genpath(fullfile(root, 'src')));
testCase.TestData.dir = fullfile(root, 'tests', 'fixtures', 'site_policy');
testCase.TestData.policy = string(fullfile(testCase.TestData.dir, 'policy_from_python.json'));
end

function testImageOfThePythonCalibrationSampleIsRefused(testCase)
paths = string(fullfile(testCase.TestData.dir, {'new_1.png'; 'cal_1.png'}));
verifyError(testCase, @() verifySitePolicy(paths, [0; 2], testCase.TestData.policy, PatientIds=["Q1"; "Q2"]), 'NetrAI:SiteSampleOverlap');
end

function testRecordedPatientsAreRequiredAndCompared(testCase)
image = string(fullfile(testCase.TestData.dir, 'new_1.png'));
verifyError(testCase, @() verifySitePolicy(image, 0, testCase.TestData.policy), 'NetrAI:SitePolicyNoProvenance');
verifyError(testCase, @() verifySitePolicy(image, 0, testCase.TestData.policy, PatientIds="P03"), 'NetrAI:SiteSampleOverlap');
verifyError(testCase, @() verifySitePolicy(image, 0, testCase.TestData.policy, PatientIds=["Q1"; "Q2"]), 'NetrAI:SiteSampleMismatch');
end

function testPolicyWithoutCalibrationImagesIsRefused(testCase)
policy = rmfield(jsondecode(fileread(testCase.TestData.policy)), 'calibration_image_sha256');
path = string(tempname) + ".json";
fid = fopen(path, 'w'); fwrite(fid, jsonencode(policy)); fclose(fid);
testCase.addTeardown(@() delete(path));
image = string(fullfile(testCase.TestData.dir, 'new_1.png'));
verifyError(testCase, @() verifySitePolicy(image, 0, path, PatientIds="Q1"), 'NetrAI:SitePolicyNoProvenance');
end

function testTextHashMatchesPython(testCase)
verifyEqual(testCase, textSHA256("P03"), "c994ecef8f751a780b28bac70bd0b73948560afbdc833e24df9c06d8b22bcfaf");
verifyEqual(testCase, textSHA256(["a"; "é"]), ["ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"; ...
    "4a99557e4033c3539de2eb65472017cad5f9557f7a0625a09f1c3f6e2ba69c4c"]);
end

function testCalibrateSiteChecksPatientIdsBeforeScoring(testCase)
out = string(tempname) + ".json";
verifyError(testCase, @() calibrateSite(["a.jpg"; "b.jpg"], [0; 2], out, PatientIds="P1", Verbose=false), 'NetrAI:SiteSampleMismatch');
verifyError(testCase, @() calibrateSite(["a.jpg"; "b.jpg"], [0; 2], out, PatientIds=["P1"; " "], Verbose=false), 'NetrAI:SiteSampleMismatch');
verifyFalse(testCase, isfile(out));
end
