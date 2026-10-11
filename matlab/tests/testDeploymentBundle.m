function tests = testDeploymentBundle
%TESTDEPLOYMENTBUNDLE Content identity checks without importing an ONNX model.
    tests = functiontests(localfunctions);
end

function setup(testCase)
    matlabDir = fileparts(fileparts(mfilename('fullpath')));
    testCase.TestData.originalPath = path;
    addpath(genpath(fullfile(matlabDir, 'src')));

    fixtureRoot = tempname;
    fixtureMatlab = fullfile(fixtureRoot, 'matlab');
    modelsDir = fullfile(fixtureMatlab, 'models');
    mkdir(modelsDir);
    mkdir(fullfile(fixtureMatlab, 'src', 'm3_grading'));
    mkdir(fullfile(fixtureMatlab, 'src', 'm4_explainability'));
    mkdir(fullfile(fixtureMatlab, 'src', 'utils'));
    roles = {'model', 'config', 'preprocessing', 'grading', 'calibration', 'routing', 'decoding'};
    paths = {'netrai_fold0.onnx', 'netrai_config.json', ...
        '../src/m3_grading/preprocessModelInput.m', ...
        '../src/m3_grading/predictDRGrade.m', ...
        '../src/m4_explainability/calibrateConfidence.m', ...
        '../src/utils/assignScreeningTriage.m','../src/utils/readScreeningImage.m'};
    contents = {'MODEL-A', '{"fixture":true}', 'PREPROCESS-A', ...
        'GRADING-A', 'CALIBRATION-A', 'ROUTING-A', 'DECODING-A'};
    artifacts = repmat(struct('role', '', 'path', '', 'sha256', ''), 1, numel(roles));
    for k = 1:numel(roles)
        artifactPath = fullfile(modelsDir, paths{k});
        writeBytes(artifactPath, contents{k});
        artifacts(k) = struct('role', roles{k}, 'path', paths{k}, ...
            'sha256', fileSHA256(artifactPath));
    end
    testCase.TestData.fixtureRoot = fixtureRoot;
    testCase.TestData.modelsDir = modelsDir;
    testCase.TestData.modelPath = fullfile(modelsDir, paths{1});
    testCase.TestData.configPath = fullfile(modelsDir, paths{2});
    testCase.TestData.bundlePath = fullfile(modelsDir, 'deployment_bundle.json');
    testCase.TestData.manifest = struct('schema_version', 2, 'artifacts', artifacts);
    testCase.TestData.contents = contents;
    writeManifest(testCase, testCase.TestData.manifest);
end

function teardown(testCase)
    path(testCase.TestData.originalPath);
    if isfolder(testCase.TestData.fixtureRoot)
        rmdir(testCase.TestData.fixtureRoot, 's');
    end
end

function testAcceptsMatchingBundle(testCase)
    [manifest, modelDigest] = verifyDeploymentBundle( ...
        testCase.TestData.modelPath, testCase.TestData.configPath, ...
        testCase.TestData.bundlePath);
    testCase.verifyEqual(manifest.schema_version, 2);
    testCase.verifyEqual(numel(manifest.artifacts), 7);
    testCase.verifyEqual(string(modelDigest), ...
        string(testCase.TestData.manifest.artifacts(1).sha256));
end

function testArtifactOrderDoesNotDetermineRoles(testCase)
    manifest = testCase.TestData.manifest;
    manifest.artifacts = manifest.artifacts([7 6 3 1 5 2 4]);
    writeManifest(testCase, manifest);
    [~, modelDigest] = verifyDeploymentBundle( ...
        testCase.TestData.modelPath, testCase.TestData.configPath, ...
        testCase.TestData.bundlePath);
    testCase.verifyEqual(string(modelDigest), ...
        string(testCase.TestData.manifest.artifacts(1).sha256));
end

function testSameLengthChangesFailForEveryRole(testCase)
    manifest = testCase.TestData.manifest;
    for k = 1:numel(manifest.artifacts)
        artifactPath = fullfile(testCase.TestData.modelsDir, manifest.artifacts(k).path);
        original = testCase.TestData.contents{k};
        changed = original;
        changed(end) = 'Z';
        writeBytes(artifactPath, changed);
        testCase.verifyEqual(numel(changed), numel(original));
        verifyBundleError(testCase, 'NetrAI:BundleIntegrity');
        writeBytes(artifactPath, original);
    end
end

function testMissingManifestFailsClosed(testCase)
    delete(testCase.TestData.bundlePath);
    verifyBundleError(testCase, 'NetrAI:BundleMissing');
end

function testMalformedJsonFailsClosed(testCase)
    writeBytes(testCase.TestData.bundlePath, '{invalid json');
    verifyBundleError(testCase, 'NetrAI:InvalidBundle');
end

function testUnsupportedSchemaFailsClosed(testCase)
    manifest = testCase.TestData.manifest;
    manifest.schema_version = 3;
    writeManifest(testCase, manifest);
    verifyBundleError(testCase, 'NetrAI:InvalidBundle');
end

function testMissingSchemaFailsClosed(testCase)
    manifest = rmfield(testCase.TestData.manifest, 'schema_version');
    writeManifest(testCase, manifest);
    verifyBundleError(testCase, 'NetrAI:InvalidBundle');
end

function testStringSchemaIsNotAccepted(testCase)
    manifest = testCase.TestData.manifest;
    manifest.schema_version = '2';
    writeManifest(testCase, manifest);
    verifyBundleError(testCase, 'NetrAI:InvalidBundle');
end

function testLegacyBundleCannotOmitDecodeContract(testCase)
    manifest = testCase.TestData.manifest;
    manifest.schema_version = 1;
    manifest.artifacts(7) = [];
    writeManifest(testCase, manifest);
    verifyBundleError(testCase,'NetrAI:InvalidBundle');
end

function testMissingRequiredRoleFailsClosed(testCase)
    manifest = testCase.TestData.manifest;
    manifest.artifacts(4) = [];
    writeManifest(testCase, manifest);
    verifyBundleError(testCase, 'NetrAI:InvalidBundle');
end

function testDuplicateRoleFailsClosed(testCase)
    manifest = testCase.TestData.manifest;
    manifest.artifacts(6).role = manifest.artifacts(5).role;
    writeManifest(testCase, manifest);
    verifyBundleError(testCase, 'NetrAI:InvalidBundle');
end

function testUnknownRoleFailsClosed(testCase)
    manifest = testCase.TestData.manifest;
    manifest.artifacts(6).role = 'unknown_source';
    writeManifest(testCase, manifest);
    verifyBundleError(testCase, 'NetrAI:InvalidBundle');
end

function testMissingDigestFailsClosed(testCase)
    manifest = testCase.TestData.manifest;
    manifest.artifacts = rmfield(manifest.artifacts, 'sha256');
    writeManifest(testCase, manifest);
    verifyBundleError(testCase, 'NetrAI:InvalidBundle');
end

function testInvalidDigestFailsClosed(testCase)
    manifest = testCase.TestData.manifest;
    manifest.artifacts(1).sha256 = repmat('g', 1, 64);
    writeManifest(testCase, manifest);
    verifyBundleError(testCase, 'NetrAI:InvalidBundle');
end

function testAbsoluteArtifactPathFailsClosed(testCase)
    manifest = testCase.TestData.manifest;
    manifest.artifacts(1).path = testCase.TestData.modelPath;
    writeManifest(testCase, manifest);
    verifyBundleError(testCase, 'NetrAI:InvalidBundle');
end

function testUnexpectedSourcePathFailsClosed(testCase)
    manifest = testCase.TestData.manifest;
    manifest.artifacts(3).path = '../src/m3_grading/unboundPreprocessor.m';
    writeManifest(testCase, manifest);
    verifyBundleError(testCase, 'NetrAI:InvalidBundle');
end

function testTraversalOutsideMatlabFailsClosed(testCase)
    manifest = testCase.TestData.manifest;
    manifest.artifacts(3).path = '../../outside.m';
    writeManifest(testCase, manifest);
    verifyBundleError(testCase, 'NetrAI:InvalidBundle');
end

function testWrongModelPathFailsClosed(testCase)
    manifest = testCase.TestData.manifest;
    manifest.artifacts(1).path = 'another_model.onnx';
    writeManifest(testCase, manifest);
    verifyBundleError(testCase, 'NetrAI:InvalidBundle');
end

function testWrongConfigPathFailsClosed(testCase)
    manifest = testCase.TestData.manifest;
    manifest.artifacts(2).path = 'another_config.json';
    writeManifest(testCase, manifest);
    verifyBundleError(testCase, 'NetrAI:InvalidBundle');
end

function testNamedModelAndConfigArgumentsAreSupported(testCase)
    manifest = testCase.TestData.manifest;
    namedModelPath = fullfile(testCase.TestData.modelsDir, 'candidate.onnx');
    namedConfigPath = fullfile(testCase.TestData.modelsDir, 'candidate_config.json');
    copyfile(testCase.TestData.modelPath, namedModelPath);
    copyfile(testCase.TestData.configPath, namedConfigPath);
    manifest.artifacts(1).path = 'candidate.onnx';
    manifest.artifacts(2).path = 'candidate_config.json';
    writeManifest(testCase, manifest);
    [~, modelDigest] = verifyDeploymentBundle( ...
        namedModelPath, namedConfigPath, testCase.TestData.bundlePath);
    testCase.verifyEqual(string(modelDigest), string(manifest.artifacts(1).sha256));
end

function writeManifest(testCase, manifest)
    writeBytes(testCase.TestData.bundlePath, jsonencode(manifest));
end

function verifyBundleError(testCase, expectedIdentifier)
    testCase.verifyError(@() verifyDeploymentBundle( ...
        testCase.TestData.modelPath, testCase.TestData.configPath, ...
        testCase.TestData.bundlePath), expectedIdentifier);
end

function writeBytes(filePath, contents)
    fid = fopen(filePath, 'wb');
    assert(fid >= 0, 'Could not create test fixture: %s', filePath);
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fwrite(fid, uint8(contents), 'uint8');
end
