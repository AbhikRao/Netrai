function report = setupDatasets(datasetPath)
%SETUPDATASETS Verify the required files in a local APTOS-style dataset.
%
% report = setupDatasets('../data/aptos2019')

    baseDir = fileparts(mfilename('fullpath'));
    if nargin < 1 || strlength(string(datasetPath)) == 0
        datasetPath = fullfile(fileparts(baseDir), 'data', 'aptos2019');
    end
    datasetPath = string(datasetPath);
    csvPath = fullfile(datasetPath, 'train.csv');
    imageDir = fullfile(datasetPath, 'train_images');
    if ~isfile(csvPath) || ~isfolder(imageDir)
        fprintf(['APTOS data was not found at %s. Download the APTOS 2019 ' ...
            'competition data from Kaggle, then place train.csv and the ' ...
            'train_images folder there.\n'], datasetPath);
        report = struct('ready',false,'dataset_path',datasetPath,'labels',0,'images',0);
        return;
    end
    labels = readtable(csvPath, 'TextType','string', 'VariableNamingRule','preserve');
    images = dir(fullfile(imageDir, '*.png'));
    report = struct('ready',true,'dataset_path',datasetPath, ...
        'labels',height(labels),'images',numel(images));
    fprintf('APTOS dataset ready: %d labels, %d PNG images at %s\n', ...
        report.labels, report.images, datasetPath);
    if report.labels ~= report.images
        warning('NetrAI:DatasetCountMismatch', 'Label/image counts differ; validation will report missing files.');
    end
end
