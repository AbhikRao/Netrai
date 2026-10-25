function tests = testMainPipelineGuards
%TESTMAINPIPELINEGUARDS Argument guards and result schema of main_pipeline that need no model weights.
tests = functiontests(localfunctions);
end

function setupOnce(~)
addpath(fullfile(fileparts(fileparts(mfilename('fullpath')))));
addpath(genpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'src')));
end

function testOnlyOneOfModelPathAndConfigPathIsRefused(testCase)
verifyError(testCase, @() main_pipeline('/nonexistent.jpg', 'ModelPath', 'x.onnx'), 'NetrAI:PartialModelPaths');
verifyError(testCase, @() main_pipeline('/nonexistent.jpg', 'ConfigPath', 'x.json'), 'NetrAI:PartialModelPaths');
end

function testMissingE768WeightsGiveAClearError(testCase)
variantOnnx = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'variant_R3_E_768', 'netrai_fold0.onnx');
if isfile(variantOnnx)
    assumeFail(testCase, 'E_768 weights are present locally.');
end
verifyError(testCase, @() main_pipeline('/nonexistent.jpg', 'Variant', 'E_768'), 'NetrAI:VariantMissing');
end

function testRejectedResultCarriesVariantAndCalibrationFields(testCase)
folder = tempname; mkdir(folder); testCase.addTeardown(@() rmdir(folder, 's'));
imwrite(zeros(256, 256, 3, 'uint8'), fullfile(folder, 'black.png'));
r = main_pipeline(fullfile(folder, 'black.png'), 'Verbose', false, 'GenerateReport', false, 'RunGradCAM', false, 'Variant', 'legacy');
verifyEqual(testCase, r.status, "rejected");
verifyTrue(testCase, isfield(r, 'model_variant') && isfield(r, 'calibration'));
verifyEqual(testCase, r.model_variant, "legacy");
[~, expected] = applySitePolicy(struct(), "");
verifyEqual(testCase, sort(fieldnames(r.calibration)), sort(fieldnames(expected)));
end
