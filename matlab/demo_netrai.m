% DEMO_NETRAI Run the trained MATLAB/ONNX pipeline on a real APTOS image.
% This demo never creates a synthetic retina or a fabricated prediction.

clc; close all;
baseDir = fileparts(mfilename('fullpath'));
projectDir = fileparts(baseDir);
imagePath = fullfile(projectDir, 'data', 'aptos2019', 'train_images', '000c1434d8d7.png');
if ~isfile(imagePath)
    imagePath = fullfile(projectDir, 'data', 'patient_sample.jpg');
end
if ~isfile(imagePath)
    error('NetrAI:DemoImageMissing', [ ...
        'No real demo image was found. Upload an RGB fundus image and call ' ...
        'main_pipeline(''your_image.png'').']);
end

fprintf('NetrAI MATLAB/ONNX demo\nInput: %s\n', imagePath);
result = main_pipeline(imagePath, ...
    'OutputDir',fullfile(baseDir, 'results', 'demo'), ...
    'Verbose',true, 'GenerateReport',true, 'EnforceQuality',false);
fprintf('Grade %d | continuous score %.4f | referable %s\n', ...
    result.grade, result.continuousScore, string(result.isReferable));
fprintf('Report: %s\n', result.reportPath);
