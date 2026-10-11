function tests=testPreprocessModelInput
%TESTPREPROCESSMODELINPUT Numerical input-contract regression fixtures.
    tests=functiontests(localfunctions);
end

function setupOnce(testCase)
    base=fileparts(fileparts(mfilename('fullpath')));
    testCase.TestData.originalPath=path;
    addpath(genpath(fullfile(base,'src')));
    testCase.TestData.config=jsondecode(fileread(fullfile(base,'models','netrai_config.json')));
    testCase.TestData.fixtures=load(fullfile(base,'tests','fixtures','preprocessing_operators.mat'));
end

function teardownOnce(testCase)
    path(testCase.TestData.originalPath);
end

function testSyntheticContractMatchesIndependentReference(testCase)
    cases=testCase.TestData.fixtures.cases;
    for k=1:numel(cases)
        item=cases{k}; config=testCase.TestData.config;
        config.input_size=double(item.target_size);
        [actual,fov]=preprocessModelInput(item.image,config);
        testCase.verifyEqual(actual,item.model_input,char(item.name));
        testCase.verifyEqual(fov,logical(item.model_fov),char(item.name));
        testCase.verifyClass(actual,'single');
    end
end

function testGrayscaleUsesTheSameRGBContract(testCase)
    image=uint8(reshape(mod(0:63*91-1,256),63,91));
    config=testCase.TestData.config; config.input_size=64;
    [grayInput,grayFov]=preprocessModelInput(image,config);
    [rgbInput,rgbFov]=preprocessModelInput(repmat(image,1,1,3),config);
    testCase.verifyEqual(grayInput,rgbInput);
    testCase.verifyEqual(grayFov,rgbFov);
end

function testFloatAndHighBitDepthAreNotSilentlyQuantized(testCase)
    config=testCase.TestData.config;
    testCase.verifyError(@()preprocessModelInput(single(ones(8,9,3)),config),'NetrAI:InvalidImage');
    testCase.verifyError(@()preprocessModelInput(uint16(ones(8,9,3)),config),'NetrAI:InvalidImage');
end

function testInvalidShapeOrEmptyInputFails(testCase)
    config=testCase.TestData.config;
    testCase.verifyError(@()preprocessModelInput(zeros(0,0,3,'uint8'),config),'NetrAI:InvalidImage');
    testCase.verifyError(@()preprocessModelInput(zeros(8,9,4,'uint8'),config),'NetrAI:InvalidImage');
end

function testUnsupportedContractFails(testCase)
    config=testCase.TestData.config;
    config.preprocessing_version='training-linear-v2';
    testCase.verifyError(@()preprocessModelInput(ones(8,9,3,'uint8'),config),'NetrAI:InvalidConfig');
end

function testInvalidConfigFailsBeforeProcessing(testCase)
    image=ones(8,9,3,'uint8');
    config=testCase.TestData.config;
    for badSize={0,1.5,NaN,Inf,[]}
        changed=config; changed.input_size=badSize{1};
        testCase.verifyError(@()preprocessModelInput(image,changed),'NetrAI:InvalidConfig');
    end
    changed=config; changed.imagenet_std=[0.1,0,0.2];
    testCase.verifyError(@()preprocessModelInput(image,changed),'NetrAI:InvalidConfig');
    changed=config; changed.ben_graham_sigma=9;
    testCase.verifyError(@()preprocessModelInput(image,changed),'NetrAI:InvalidConfig');
    changed=rmfield(config,'black_crop_threshold');
    testCase.verifyError(@()preprocessModelInput(image,changed),'NetrAI:InvalidConfig');
end
