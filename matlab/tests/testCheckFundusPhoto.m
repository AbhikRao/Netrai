function tests = testCheckFundusPhoto
%TESTCHECKFUNDUSPHOTO The non-fundus input guard: same scores as the Python guard (fixture also checked by
% python/tests/test_fundus_guard.py), a fundus photograph passes, and main_pipeline rejects or (as shipped) flags a photograph of peppers.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
addpath(genpath(fullfile(root, 'src')));
assumeTrue(testCase, onnxImporterInstalled(), 'The ONNX converter support package is not installed.');
testCase.TestData.root = root;
testCase.TestData.fixture = jsondecode(fileread(fullfile(root, 'tests', 'fixtures', 'fundus_guard_scores.json')));
testCase.TestData.sample = readScreeningImage(string(fullfile(root, '..', 'data', 'patient_sample.jpg')));
end

function testScoresEqualThePythonGuard(testCase)
fundus = checkFundusPhoto(testCase.TestData.sample);
verifyTrue(testCase, fundus.is_fundus);
verifyEqual(testCase, fundus.score, testCase.TestData.fixture.patient_sample, 'AbsTol', 1e-3);
peppers = checkFundusPhoto(imread('peppers.png'));
verifyFalse(testCase, peppers.is_fundus);
verifyEqual(testCase, peppers.score, testCase.TestData.fixture.peppers, 'RelTol', 1e-4);
end

function testPipelineRejectsOrFlagsANonFundusPhoto(testCase)
folder = tempname; mkdir(folder); testCase.addTeardown(@() rmdir(folder, 's'));
imwrite(imread('peppers.png'), fullfile(folder, 'peppers.png'));
r = main_pipeline(fullfile(folder, 'peppers.png'), 'Variant', 'legacy', 'Verbose', false, 'GenerateReport', false, 'RunGradCAM', false);
guard = checkFundusPhoto(imread('peppers.png'));
if guard.action == "reject"
    verifyEqual(testCase, [r.status r.rejectionReason r.triageAction], ["rejected" "not_fundus" "recapture"]);
    verifyTrue(testCase, isnan(r.grade));
else
    verifyEqual(testCase, r.status, "success");
    verifyNotEmpty(testCase, r.inputWarning);
end
end

function testBatchCsvCarriesTheGuard(testCase)
folder = tempname; mkdir(folder); testCase.addTeardown(@() rmdir(folder, 's'));
imwrite(imread('peppers.png'), fullfile(folder, 'peppers.png'));
T = batchProcess(folder, fullfile(folder, 'batch.csv'));
verifyEqual(testCase, T.fundus_guard_score, testCase.TestData.fixture.peppers, 'RelTol', 1e-4);
if checkFundusPhoto(imread('peppers.png')).action == "flag"
    verifyEqual(testCase, T.status, "success");
    verifyNotEqual(testCase, T.input_warning, "");
else
    verifyEqual(testCase, T.status, "rejected");
end
end

function testTamperedParametersAreRefused(testCase)
root = testCase.TestData.root;
folder = tempname; mkdir(folder); testCase.addTeardown(@() rmdir(folder, 's'));
for name = ["fundus_guard.json" "quality_model_candidate.onnx" "quality_model_candidate.json"]
    copyfile(fullfile(root, 'models', name), fullfile(folder, name));
end
fid = fopen(fullfile(root, 'models', 'fundus_guard.f32'), 'r'); blob = fread(fid, inf, '*uint8'); fclose(fid);
blob(2401) = bitxor(blob(2401), uint8(1));
fid = fopen(fullfile(folder, 'fundus_guard.f32'), 'w'); fwrite(fid, blob, 'uint8'); fclose(fid);
verifyError(testCase, @() checkFundusPhoto(testCase.TestData.sample, fullfile(folder, 'fundus_guard.json')), 'NetrAI:FundusGuardIntegrity');
end

function testCachedNetworkSurvivesAPathChangeAndCd(testCase)
% The import writes custom-layer classes into matlab/onnx_layers; a cached network used to fail after that folder left the path
% (a test runner restores the path after its tests) or after a cd ("Execution failed during layers ...").
first = checkFundusPhoto(testCase.TestData.sample);
rmpath(onnxLayerFolder());
previous = cd(tempdir); testCase.addTeardown(@() cd(previous));
again = checkFundusPhoto(testCase.TestData.sample);
verifyEqual(testCase, again.score, first.score, 'AbsTol', 1e-12);
end

function tf = onnxImporterInstalled()
try
    addons = matlab.addons.installedAddons;
    tf = exist('importNetworkFromONNX', 'file') == 2 && any(contains(addons.Name, "Converter for ONNX"));
catch
    tf = false;
end
end
