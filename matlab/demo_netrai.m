function result = demo_netrai(imagePath)
%DEMO_NETRAI Run a quality-gated MATLAB screening demonstration.
% Supply a permitted local RGB fundus image. Reports use a fresh directory.

    arguments
        imagePath (1,1) string = ""
    end
    baseDir = fileparts(mfilename('fullpath'));
    projectDir = fileparts(baseDir);
    if strlength(imagePath) == 0
        imagePath = string(fullfile(projectDir, 'data', 'aptos2019', ...
            'train_images', '000c1434d8d7.png'));
        if ~isfile(imagePath)
            imagePath = string(fullfile(projectDir, 'data', 'patient_sample.jpg'));
        end
    end
    if ~isfile(imagePath)
        error('NetrAI:DemoImageMissing', ...
            'Supply a permitted local image: result = demo_netrai("fundus.png").');
    end
    demoRoot = fullfile(baseDir, 'results', 'demo');
    if ~isfolder(demoRoot), mkdir(demoRoot); end
    outputDir = tempname(demoRoot);
    fprintf('NetrAI MATLAB screening prototype — research demonstration\n');
    result = main_pipeline(imagePath, OutputDir=outputDir, ...
        Verbose=true, GenerateReport=true, RunGradCAM=true, EnforceQuality=true);
    if result.status == "rejected"
        fprintf('Recapture requested: %s\n', result.qualityFeedback);
    else
        fprintf('Grade %d | routing %s\n', result.grade, result.triageAction);
        fprintf('Report: %s\n', result.reportPath);
    end
end
