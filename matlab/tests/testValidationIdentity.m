function tests = testValidationIdentity
%TESTVALIDATIONIDENTITY Cohort hashes bind the bytes used for image decoding.
    tests = functiontests(localfunctions);
end

function setup(testCase)
    matlabDir = fileparts(fileparts(mfilename('fullpath')));
    testCase.TestData.originalPath = path;
    addpath(genpath(fullfile(matlabDir, 'src')));
    fixtureRoot = tempname;
    mkdir(fixtureRoot);
    originalPath = fullfile(fixtureRoot, 'source.png');
    [x, y] = meshgrid(uint8(0:31), uint8(0:23));
    image = cat(3, x .* uint8(7), y .* uint8(9), x + y);
    imwrite(image, originalPath);
    testCase.TestData.fixtureRoot = fixtureRoot;
    testCase.TestData.originalPathToImage = originalPath;
    testCase.TestData.image = image;
    testCase.TestData.expectedDigest = fileSHA256(originalPath);
end

function teardown(testCase)
    path(testCase.TestData.originalPath);
    if isfolder(testCase.TestData.fixtureRoot)
        rmdir(testCase.TestData.fixtureRoot, 's');
    end
end

function testStagesAndDecodesExactSnapshotBytes(testCase)
    source = testCase.TestData.originalPathToImage;
    [image, boundPath, cleanup] = readBoundValidationImage( ...
        source, testCase.TestData.expectedDigest); %#ok<ASGLU>
    testCase.verifyEqual(image, testCase.TestData.image);
    testCase.verifyNotEqual(string(boundPath), string(source));
    testCase.verifyTrue(isfile(boundPath));
    testCase.verifyEqual(fileSHA256(boundPath), testCase.TestData.expectedDigest);
    testCase.verifyEqual(readBytes(boundPath), readBytes(source));
    [~, ~, extension] = fileparts(boundPath);
    testCase.verifyEqual(string(extension), ".png");
end

function testSameLengthMutationIsRejectedBeforeDecoding(testCase)
    source = testCase.TestData.originalPathToImage;
    bytes = readBytes(source);
    originalLength = numel(bytes);
    bytes(end) = bitxor(bytes(end), uint8(1));
    writeBytes(source, bytes);
    testCase.verifyEqual(numel(readBytes(source)), originalLength);
    testCase.verifyError(@() readBoundValidationImage( ...
        source, testCase.TestData.expectedDigest), 'NetrAI:ChangedInput');
end

function testStagedInputSurvivesOriginalReplacement(testCase)
    source = testCase.TestData.originalPathToImage;
    [~, boundPath, cleanup] = readBoundValidationImage( ...
        source, testCase.TestData.expectedDigest); %#ok<ASGLU>
    imwrite(zeros(size(testCase.TestData.image), 'uint8'), source);
    testCase.verifyNotEqual(fileSHA256(source), testCase.TestData.expectedDigest);
    testCase.verifyEqual(fileSHA256(boundPath), testCase.TestData.expectedDigest);
    testCase.verifyEqual(imread(boundPath), testCase.TestData.image);
    testCase.verifyError(@() readBoundValidationImage( ...
        source, testCase.TestData.expectedDigest), 'NetrAI:ChangedInput');
end

function testCleanupRemovesOnlyStagedFile(testCase)
    source = testCase.TestData.originalPathToImage;
    [~, boundPath, cleanup] = readBoundValidationImage( ...
        source, testCase.TestData.expectedDigest);
    testCase.verifyTrue(isfile(boundPath));
    clear cleanup;
    testCase.verifyFalse(isfile(boundPath));
    testCase.verifyTrue(isfile(source));
    testCase.verifyEqual(fileSHA256(source), testCase.TestData.expectedDigest);
end

function testSchemaThreeCapturesPolicyFromBoundConfigBytes(testCase)
    [matlabDir,modelPath,configPath,labelsPath,cohort,configJson]=identityFixture(testCase);
    [identity,paths]=buildValidationIdentity(matlabDir,modelPath,configPath, ...
        labelsPath,"","model",cohort);
    testCase.verifyEqual(identity.schema_version,3);
    testCase.verifyEqual(identity.config_json,configJson);
    testCase.verifyEqual(identity.config_sha256, ...
        sha256Bytes(unicode2native(identity.config_json,'UTF-8')));
    testCase.verifyEqual(identity.decision_policy.grade_thresholds,[0.5 1.5 2.5 3.5]);
    testCase.verifyEqual(identity.decision_policy.referable_head_threshold,0.5);
    testCase.verifyEqual(identity.decision_policy.confidence_temperature,0.8);
    testCase.verifyEqual(identity.expected_cohort.image_sha256,testCase.TestData.expectedDigest);
    testCase.verifyTrue(isfile(paths(1)));
end

function testPolicyCaptureRejectsChangedConfiguration(testCase)
    [matlabDir,modelPath,configPath,labelsPath,cohort,configJson]=identityFixture(testCase);
    changed=strrep(configJson,'"confidence_temperature":0.8','"confidence_temperature":0.9');
    testCase.verifyEqual(numel(changed),numel(configJson));
    testCase.verifyNotEqual(changed,configJson);
    writeBytes(configPath,unicode2native(changed,'UTF-8'));
    testCase.verifyError(@() buildValidationIdentity(matlabDir,modelPath,configPath, ...
        labelsPath,"","model",cohort),'NetrAI:BundleIntegrity');
end

function testPolicyCaptureRejectsInvalidBoundThresholds(testCase)
    config=struct('grade_thresholds',[0.5 1.5 1.5 3.5], ...
        'referable_head_threshold',0.5,'confidence_temperature',0.8);
    [matlabDir,modelPath,configPath,labelsPath,cohort]=identityFixture(testCase,config);
    testCase.verifyError(@() buildValidationIdentity(matlabDir,modelPath,configPath, ...
        labelsPath,"","model",cohort),'NetrAI:InvalidDecisionPolicy');
end

function [matlabDir,modelPath,configPath,labelsPath,cohort,configJson]=identityFixture(testCase,config)
    if nargin<2
        config=struct('grade_thresholds',[0.5 1.5 2.5 3.5], ...
            'referable_head_threshold',0.5,'confidence_temperature',0.8);
    end
    matlabDir=fullfile(testCase.TestData.fixtureRoot,'matlab');
    modelsDir=fullfile(matlabDir,'models'); mkdir(modelsDir);
    paths=["netrai_fold0.onnx","netrai_config.json", ...
        "../src/m3_grading/preprocessModelInput.m", ...
        "../src/m3_grading/predictDRGrade.m", ...
        "../src/m4_explainability/calibrateConfidence.m", ...
        "../src/utils/assignScreeningTriage.m","../src/utils/readScreeningImage.m"];
    roles=["model","config","preprocessing","grading","calibration","routing","decoding"];
    configJson=jsonencode(config);
    artifacts=repmat(struct('role','','path','','sha256',''),numel(paths),1);
    for index=1:numel(paths)
        artifactPath=fullfile(modelsDir,paths(index));
        parent=fileparts(artifactPath); if ~isfolder(parent), mkdir(parent); end
        contents=uint8('synthetic identity fixture');
        if roles(index)=="config", contents=unicode2native(configJson,'UTF-8'); end
        writeBytes(artifactPath,contents);
        artifacts(index)=struct('role',char(roles(index)),'path',char(paths(index)), ...
            'sha256',fileSHA256(artifactPath));
    end
    writeBytes(fullfile(modelsDir,'netrai_bundle.json'), ...
        unicode2native(jsonencode(struct('schema_version',2,'artifacts',artifacts)),'UTF-8'));
    modelPath=fullfile(modelsDir,paths(1)); configPath=fullfile(modelsDir,paths(2));
    datasetDir=fullfile(testCase.TestData.fixtureRoot,'dataset');
    mkdir(fullfile(datasetDir,'train_images'));
    copyfile(testCase.TestData.originalPathToImage,fullfile(datasetDir,'train_images','fixture.png'));
    cohort=table("fixture",0,'VariableNames',{'id_code','diagnosis'});
    labelsPath=fullfile(datasetDir,'train.csv'); writetable(cohort,labelsPath);
end

function bytes = readBytes(filePath)
    fid = fopen(filePath, 'rb');
    assert(fid >= 0, 'Could not read test fixture: %s', filePath);
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    bytes = fread(fid, inf, '*uint8');
end

function writeBytes(filePath, bytes)
    fid = fopen(filePath, 'wb');
    assert(fid >= 0, 'Could not write test fixture: %s', filePath);
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fwrite(fid, bytes, 'uint8');
end
