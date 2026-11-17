function tests = testNetraiApp
%TESTNETRAIAPP The demonstration console shows what the pipeline returned, for each routing outcome, without running a model.
% A stub with main_pipeline's interface supplies the results. Skipped without a display (CI). On Linux, MATLAB R2026a started with
% -batch can end with a fatal error at shutdown once any uifigure has been created (a bare uifigure does it too); the test
% results are complete before that.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root); addpath(genpath(fullfile(root, 'src')));
testCase.TestData.image = [tempname '.png'];
imwrite(uint8(repmat(reshape([120 60 30], 1, 1, 3), 48, 64)), testCase.TestData.image);
testCase.addTeardown(@() delete(testCase.TestData.image));
assumeTrue(testCase, isempty(getenv('GITHUB_ACTIONS')) && (ispc || ismac || ~isempty(getenv('DISPLAY'))), ...
    'No display (or a CI runner): app windows are not created here.');
end

function testReferralIsShownWithItsWarnings(testCase)
app = openApp(testCase, @(path, varargin) gradedResult("refer", 3, "UNCALIBRATED: frozen thresholds", varargin));
r = app.screen(testCase.TestData.image);
verifyEqual(testCase, string(app.Banner.Text), "REFER");
verifyEqual(testCase, string(app.Details(1).Text), "3 of 4 (refer from 2)");
verifyEqual(testCase, string(app.Details(2).Text), "Severe NPDR (70%)");
verifyTrue(testCase, contains(string(app.Notices.Text), "not calibrated"));           % the uncalibrated cut-off is said plainly
verifyTrue(testCase, contains(string(app.Details(6).Text), "not calibrated"));
verifyEqual(testCase, string(r.triageAction), "refer");
s = app.state();
verifyEqual(testCase, s.view, "attention");                                             % the attention map is shown first
end

function testNoReferralIsNotCalledAClearance(testCase)
app = openApp(testCase, @(path, varargin) gradedResult("auto_clear", 0, "site-calibrated (not independently validated)", varargin));
app.screen(testCase.TestData.image);
verifyEqual(testCase, string(app.Banner.Text), "NO REFERRAL");
verifyTrue(testCase, contains(string(app.BannerNote.Text), "not a clearance"));
verifyFalse(testCase, contains(string(app.Notices.Text), "not calibrated"));
verifyEqual(testCase, string(app.Details(6).Text), "Calibrated for this site");
end

function testDisagreementGoesToAPerson(testCase)
app = openApp(testCase, @(path, varargin) gradedResult("uncertain_human_review", 2, "UNCALIBRATED: frozen thresholds", varargin));
app.screen(testCase.TestData.image);
verifyEqual(testCase, string(app.Banner.Text), "HUMAN REVIEW");
end

function testRejectedImageShowsFeedbackAndNoGrade(testCase)
app = openApp(testCase, @(path, varargin) struct('status', "rejected", 'rejectionReason', "image_quality", 'iqs', 0.21, ...
    'qualityFeedback', "Refocus and retake the photograph.", 'triageAction', "recapture", 'model_variant', "legacy", ...
    'grade', NaN, 'reportPath', ""));
app.screen(testCase.TestData.image);
verifyEqual(testCase, string(app.Banner.Text), "RECAPTURE");
verifyEqual(testCase, string(app.BannerNote.Text), "Refocus and retake the photograph.");
verifyEqual(testCase, string(app.Details(1).Text), "not graded");
verifyEqual(testCase, string(app.ReportButton.Enable), "off");
s = app.state();
verifyEqual(testCase, s.view, "photo");
end

function testNonFundusWarningIsShown(testCase)
app = openApp(testCase, @(path, varargin) setfield(gradedResult("refer", 2, "UNCALIBRATED: frozen thresholds", varargin), ...
    'inputWarning', "This image does not look like a fundus (retina) photograph; capture a fundus image of the eye.")); %#ok<SFLD>
app.screen(testCase.TestData.image);
verifyTrue(testCase, contains(string(app.Notices.Text), "does not look like a fundus photograph"));
end

function testPipelineErrorIsNotReplacedByAResult(testCase)
app = openApp(testCase, @failingPipeline);
verifyError(testCase, @() app.screen(testCase.TestData.image), 'NetrAI:Stub');
verifyEqual(testCase, string(app.Banner.Text), "NO RESULT YET");
verifyTrue(testCase, contains(string(app.Status.Text), "model failed"));
end

function testPipelineReceivesTheChosenOptions(testCase)
app = openApp(testCase, @(path, varargin) gradedResult("refer", 2, "UNCALIBRATED: frozen thresholds", varargin));
app.VariantDropDown.Value = 'legacy';
r = app.screen(testCase.TestData.image);
verifyEqual(testCase, string(r.received.Variant), "legacy");
verifyTrue(testCase, r.received.EnforceQuality && r.received.RunGradCAM && ~r.received.Verbose);
verifyFalse(testCase, isfield(r.received, 'SitePolicy'));                               % none chosen, none passed
end

function testCapacityTableFollowsTheRecordedSweep(testCase)
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
csvPath = fullfile(root, 'results', 'benchmarks', 'matlab_workflow_2026-09-28', 'simulink_scenario_sweep.csv');
assumeTrue(testCase, isfile(csvPath));
app = openApp(testCase, @failingPipeline);
app.ReviewersDropDown.Value = 1; app.RejectDropDown.Value = 0.10; app.showCapacity();
data = app.CapacityTable.Data;
verifyEqual(testCase, string(data(:, 1))', ["0.25" "1" "5"]);
verifyEqual(testCase, cell2mat(data(:, 2))', [32142 99893 99904]);                      % the published rows
verifyEqual(testCase, string(data(:, 6))', ["32.14" "99.89" "99.90"]);
end

function r = failingPipeline(varargin) %#ok<STOUT>
error('NetrAI:Stub', 'model failed');
end

function app = openApp(testCase, pipeline)
app = netraiApp(Visible="off", Pipeline=pipeline);
testCase.addTeardown(@() delete(app.Figure(isvalid(app.Figure))));
end

function r = gradedResult(triage, grade, calibrationStatus, received)
probs = 0.075 * ones(1, 5); probs(grade + 1) = 0.70;
[yy, xx] = ndgrid(linspace(-1, 1, 32));
r = struct('status', "success", 'grade', grade, 'probabilities', probs, 'referableHeadProbability', 0.81, 'iqs', 0.74, ...
    'triageAction', string(triage), 'gradcamStatus', "available", 'model_variant', "E_768", 'inputWarning', "", ...
    'calibration', struct('calibration_status', string(calibrationStatus)), 'reportPath', "", ...
    'visuals', struct('analysisImage', zeros(32, 32, 3), 'overlayImage', zeros(32, 32, 3), 'modelDisplay', 0.4 * ones(32, 32, 3), ...
        'heatmap', exp(-(xx .^ 2 + yy .^ 2) * 4)), ...
    'received', cell2struct(received(2:2:end)', string(received(1:2:end))', 1));
end
