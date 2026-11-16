function tests = testReferralGradCAM
%TESTREFERRALGRADCAM The closed-form referral Grad-CAM matches MATLAB's autodiff gradCAM on each member (synthetic networks).
% One member (the legacy structure) and two members averaged like the E_768 graph, each with the NetrAI head (conv -> SiLU -> global
% pooling -> Linear 512 - BN - SiLU - Linear 256 - BN - SiLU -> referral Linear 1 / grade Linear 5). No model weights needed for
% these; the two real-model tests compare with fixtures that python/tests also checks (the same map in both runtimes).
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
assumeFalse(testCase, isempty(ver('nnet')), 'Deep Learning Toolbox is not installed.');
addpath(genpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'src')));
rng(15);                                     % a seed where every member has a non-degenerate map
testCase.TestData.net = memberNetwork(2);
testCase.TestData.single = memberNetwork(1);
testCase.TestData.x = rand(64, 64, 3, 'single');
end

function testMemberMapsEqualAutodiffGradCAM(testCase)
[heatmap, check, members, coarse] = referralGradCAM(testCase.TestData.net, testCase.TestData.x);
verifyEqual(testCase, check.members, 2);
verifyLessThan(testCase, check.max_logit_error, 1e-4);              % recomputed referral logits = network output
verifyCoarseMapsEqualAutodiff(testCase, testCase.TestData.net, coarse);
c = corrcoef(reshape(members(:, :, 1), [], 1), reshape(members(:, :, 2), [], 1));
verifyLessThan(testCase, c(1, 2), 0.99);                            % the two members explain differently here
verifyEqual(testCase, size(heatmap), [64 64]);
verifyEqual(testCase, [min(heatmap(:)) max(heatmap(:))], [0 1], 'AbsTol', 1e-6);
both = mean(members, 3); both = (both - min(both(:))) / (max(both(:)) - min(both(:)) + 1e-7);
verifyEqual(testCase, heatmap, both, 'AbsTol', 1e-9);                 % the average of the two members, not one of them
end

function testSingleMemberEqualsAutodiffGradCAM(testCase)
[heatmap, check, members, coarse] = referralGradCAM(testCase.TestData.single, testCase.TestData.x);
verifyEqual(testCase, check.members, 1);
verifyLessThan(testCase, check.max_logit_error, 1e-4);
verifyCoarseMapsEqualAutodiff(testCase, testCase.TestData.single, coarse);
verifyEqual(testCase, heatmap, (members - min(members(:))) / (max(members(:)) - min(members(:)) + 1e-7), 'AbsTol', 1e-9);
end

function testGenerateGradCAMUsesTheReferralGradCAM(testCase)
for net = {testCase.TestData.net, testCase.TestData.single}
    [heatmap, status] = generateGradCAM(net{1}, testCase.TestData.x, 6);
    verifyEqual(testCase, status, "available");
    verifyEqual(testCase, heatmap, referralGradCAM(net{1}, testCase.TestData.x), 'AbsTol', 1e-12);
end
end

function testRecordedForwardPassGivesTheSameOutputsAndTheSameMap(testCase)
% ForwardRecorder (used by main_pipeline for the grading call): the six outputs are those of a plain predict, bit for bit, and
% the map built from the recorded activations is the map referralGradCAM computes with its own forward pass.
for net = {testCase.TestData.net, testCase.TestData.single}
    dlX = dlarray(testCase.TestData.x, 'SSCB');
    recorder = ForwardRecorder(net{1});
    verifyFalse(testCase, recorder.hasForward());
    output = predict(recorder, dlX);
    verifyEqual(testCase, extractdata(output), extractdata(predict(net{1}, dlX)));          % exact
    verifyTrue(testCase, recorder.hasForward());
    verifyEqual(testCase, recorder.ForwardPasses, 1);
    [own, ownCheck] = referralGradCAM(net{1}, testCase.TestData.x);
    [shared, sharedCheck] = referralGradCAM(net{1}, testCase.TestData.x, recorder);
    verifyEqual(testCase, shared, own);                                                     % exact
    verifyEqual(testCase, [ownCheck.forward_passes sharedCheck.forward_passes], [1 0]);
    [viaGenerate, status] = generateGradCAM(net{1}, testCase.TestData.x, 6, recorder);
    verifyEqual(testCase, status, "available");
    verifyEqual(testCase, viaGenerate, own);
    verifyEqual(testCase, recorder.ForwardPasses, 1);                                       % no second pass anywhere
end
end

function testRecordedForwardPassOfAnotherInputIsRefused(testCase)
recorder = ForwardRecorder(testCase.TestData.single);
predict(recorder, dlarray(testCase.TestData.x, 'SSCB'));
other = testCase.TestData.x; other(1) = other(1) + 0.5;
verifyError(testCase, @() referralGradCAM(testCase.TestData.single, other, recorder), 'NetrAI:GradCAMRecordedInputMismatch');
[heatmap, status] = generateGradCAM(testCase.TestData.single, other, 6, recorder);          % no map from the wrong pass
verifyEmpty(testCase, heatmap);
verifyTrue(testCase, startsWith(status, "unavailable"));
end

function testGradingThroughTheRecorderEqualsGradingThroughTheNetwork(testCase)
% predictDRGrade (content-bound, unchanged) gives the same nine outputs whether it is handed the network or the recorder.
thresholds = [0.5 1.5 2.5 3.5];
x = rand(256, 256, 3, 'single');                     % predictDRGrade accepts 256-1024 pixel inputs
plain = cell(1, 9); recorded = cell(1, 9);
[plain{:}] = predictDRGrade(testCase.TestData.single, x, thresholds, 0.5);
recorder = ForwardRecorder(testCase.TestData.single);
[recorded{:}] = predictDRGrade(recorder, x, thresholds, 0.5);
verifyEqual(testCase, recorded, plain);                                                     % exact
verifyTrue(testCase, recorder.hasForward());
end

function testNetworkWithoutTheNetrAIHeadIsRefused(testCase)
lg = layerGraph([imageInputLayer([16 16 3], 'Normalization', 'none', 'Name', 'in'); convolution2dLayer(3, 4, 'Name', 'c')
    globalAveragePooling2dLayer('Name', 'g'); fullyConnectedLayer(6, 'Name', 'fc')]);
net = dlnetwork(lg);
x = rand(16, 16, 3, 'single');
verifyError(testCase, @() referralGradCAM(net, x), 'NetrAI:GradCAMHeadMismatch');
verifyError(testCase, @() ForwardRecorder(net), 'NetrAI:GradCAMHeadMismatch');
[heatmap, status] = generateGradCAM(net, x, 6);                     % no substitute map
verifyEmpty(testCase, heatmap);
verifyTrue(testCase, startsWith(status, "unavailable"));
end

function testRealLegacyMapMatchesTheCrossRuntimeFixture(testCase)
% 8 x 8 block means of the legacy map of data/patient_sample.jpg; python/tests/test_head_gradcam.py checks the Python map against it.
root = fileparts(fileparts(mfilename('fullpath')));
verifyRealModelFixture(testCase, fullfile(root, 'models'), 'legacy_gradcam_patient_sample_64.f32', 64);
end

function testRealE768MapMatchesTheCrossRuntimeFixture(testCase)
% 8 x 8 block means of the E_768 map of data/patient_sample.jpg; python/tests/test_variant_e768.py checks the Python map against it.
root = fileparts(fileparts(mfilename('fullpath')));
modelDir = fullfile(root, 'variant_R3_E_768');
assumeTrue(testCase, isfile(fullfile(modelDir, 'netrai_fold0.onnx')), 'E_768 weights are not installed.');
verifyRealModelFixture(testCase, modelDir, 'e768_gradcam_patient_sample_96.f32', 96);
end

function verifyCoarseMapsEqualAutodiff(testCase, net, coarse)
% Autodiff Grad-CAM of the network output at member k's last activation, before upsampling: equal up to the positive factor
% 1/(members x feature-map area) that min-max scaling removes.
dlX = dlarray(testCase.TestData.x, 'SSCB');
for k = 1:numel(coarse)
    ref = double(extractdata(gradCAM(net, dlX, 6, 'FeatureLayer', "m" + k + "_silu", ...
        'ReductionLayer', net.OutputNames{1}, 'OutputUpsampling', 'none')));
    verifyGreaterThan(testCase, max(coarse{k}(:)), 0);              % a non-degenerate map for this seed
    verifyEqual(testCase, coarse{k} / max(coarse{k}(:)), ref / max(ref(:)), 'AbsTol', 1e-4);
end
end

function verifyRealModelFixture(testCase, modelDir, fixtureName, blocks)
assumeTrue(testCase, onnxImporterInstalled(), 'The ONNX converter support package is not installed.');
root = fileparts(fileparts(mfilename('fullpath')));
[net, config] = loadNetrAIModel(string(fullfile(modelDir, 'netrai_fold0.onnx')), string(fullfile(modelDir, 'netrai_config.json')));
x = preprocessModelInput(readScreeningImage(string(fullfile(root, '..', 'data', 'patient_sample.jpg'))), config);
[heatmap, status] = generateGradCAM(net, x, 6);
verifyEqual(testCase, status, "available");
means = squeeze(mean(mean(reshape(heatmap, 8, blocks, 8, blocks), 1), 3));
fid = fopen(fullfile(root, 'tests', 'fixtures', fixtureName), 'r');
fixture = reshape(fread(fid, blocks * blocks, 'float32=>double', 0, 'ieee-le'), blocks, blocks).';
fclose(fid);
verifyEqual(testCase, means, fixture, 'AbsTol', 1e-4);
end

function tf = onnxImporterInstalled()
try
    addons = matlab.addons.installedAddons;
    tf = exist('importNetworkFromONNX', 'file') == 2 && any(contains(addons.Name, "Converter for ONNX"));
catch
    tf = false;
end
end

function net = memberNetwork(members)
lg = layerGraph(imageInputLayer([64 64 3], 'Normalization', 'none', 'Name', 'in'));
if members > 1
    lg = addLayers(lg, [additionLayer(members, 'Name', 'sum'); functionLayer(@(X) X / members, 'Formattable', true, 'Name', 'mean')]);
end
for k = 1:members
    m = "m" + k + "_";
    lg = addLayers(lg, convolution2dLayer(3, 16, 'Stride', 4, 'Padding', 'same', 'Name', m + "conv", ...
        'Weights', 0.4 * randn(3, 3, 3, 16), 'Bias', 0.1 * randn(1, 1, 16)));
    lg = silu(lg, m + "conv", m + "silu");
    lg = addLayers(lg, [globalAveragePooling2dLayer('Name', m + "pool")
        fullyConnectedLayer(512, 'Name', m + "fc1", 'Weights', 0.2 * randn(512, 16), 'Bias', 0.1 * randn(512, 1)); bn(m + "bn1", 512)]);
    lg = connectLayers(lg, m + "silu", m + "pool");
    lg = silu(lg, m + "bn1", m + "a1");
    lg = addLayers(lg, [fullyConnectedLayer(256, 'Name', m + "fc2", 'Weights', 0.05 * randn(256, 512), 'Bias', 0.1 * randn(256, 1)); bn(m + "bn2", 256)]);
    lg = connectLayers(lg, m + "a1", m + "fc2");
    lg = silu(lg, m + "bn2", m + "a2");
    lg = addLayers(lg, [fullyConnectedLayer(5, 'Name', m + "grade", 'Weights', 0.1 * randn(5, 256), 'Bias', zeros(5, 1))
        concatenationLayer(1, 2, 'Name', m + "out")]);   % fully connected outputs are C x B
    lg = addLayers(lg, fullyConnectedLayer(1, 'Name', m + "ref", 'Weights', 0.1 * randn(1, 256), 'Bias', 0.2));
    lg = connectLayers(lg, m + "a2", m + "grade");
    lg = connectLayers(lg, m + "a2", m + "ref");
    lg = connectLayers(lg, m + "ref", m + "out/in2");
    lg = connectLayers(lg, "in", m + "conv");
    if members > 1
        lg = connectLayers(lg, m + "out", "sum/in" + k);
    end
end
net = dlnetwork(lg);
end

function lg = silu(lg, source, name)
% x -> x * sigmoid(x), as the ONNX export represents SiLU.
lg = addLayers(lg, sigmoidLayer('Name', name + "_sig"));
lg = addLayers(lg, multiplicationLayer(2, 'Name', name));
lg = connectLayers(lg, source, name + "_sig");
lg = connectLayers(lg, source, name + "/in1");
lg = connectLayers(lg, name + "_sig", name + "/in2");
end

function layer = bn(name, n)
layer = batchNormalizationLayer('Name', name, 'TrainedMean', 0.1 * randn(1, 1, n), 'TrainedVariance', 0.5 + rand(1, 1, n), ...
    'Offset', 0.1 * randn(1, 1, n), 'Scale', 1 + 0.1 * randn(1, 1, n));
end
