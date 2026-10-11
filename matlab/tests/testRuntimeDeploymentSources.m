function tests=testRuntimeDeploymentSources
% Synthetic alternate deployments exercise resolution without ONNX imports.
    tests=functiontests(localfunctions);
end

function setup(testCase)
    matlabDir=fileparts(fileparts(mfilename('fullpath')));
    testCase.TestData.originalPath=path;
    addpath(genpath(fullfile(matlabDir,'src')));
    testCase.TestData.root=tempname; mkdir(testCase.TestData.root);
    testCase.TestData.roles=["preprocessing","grading","calibration","routing","decoding"];
    testCase.TestData.functions=["preprocessModelInput","predictDRGrade","calibrateConfidence", ...
        "assignScreeningTriage","readScreeningImage"];
    artifacts=repmat(struct('role','','sha256',''),5,1);
    sources=strings(5,1);
    for index=1:5
        sources(index)=string(which(testCase.TestData.functions(index)));
        artifacts(index)=struct('role',char(testCase.TestData.roles(index)), ...
            'sha256',fileSHA256(sources(index)));
    end
    testCase.TestData.sources=sources;
    testCase.TestData.manifest=struct('artifacts',artifacts);
end

function teardown(testCase)
    path(testCase.TestData.originalPath);
    rehash;
    if isfolder(testCase.TestData.root), rmdir(testCase.TestData.root,'s'); end
end

function testAcceptsActuallyResolvedSourceBytes(testCase)
    resolved=verifyRuntimeDeploymentSources(testCase.TestData.manifest);
    verifyEqual(testCase,string({resolved.role}),testCase.TestData.roles);
    verifyEqual(testCase,string({resolved.sha256}), ...
        string({testCase.TestData.manifest.artifacts.sha256}));
end

function testRejectsChangedShadowForEveryRuntimeRole(testCase)
    for index=1:5
        folder=fullfile(testCase.TestData.root,sprintf('shadow_%d',index)); mkdir(folder);
        target=fullfile(folder,testCase.TestData.functions(index)+".m");
        copyfile(testCase.TestData.sources(index),target);
        appendComment(target);
        addpath(folder,'-begin'); rehash;
        verifyEqual(testCase,string(which(testCase.TestData.functions(index))),string(target));
        verifyError(testCase,@() verifyRuntimeDeploymentSources(testCase.TestData.manifest), ...
            'NetrAI:RuntimeSourceMismatch');
        rmpath(folder); rehash;
    end
end

function testByteIdenticalRelocatedSourcesRemainValid(testCase)
    folder=fullfile(testCase.TestData.root,'identical'); mkdir(folder);
    for index=1:5
        copyfile(testCase.TestData.sources(index), ...
            fullfile(folder,testCase.TestData.functions(index)+".m"));
    end
    addpath(folder,'-begin'); rehash;
    resolved=verifyRuntimeDeploymentSources(testCase.TestData.manifest);
    for index=1:5
        verifyEqual(testCase,string(resolved(index).path), ...
            string(fullfile(folder,testCase.TestData.functions(index)+".m")));
    end
end

function testReboundUnexecutedSiblingCodeCannotAuthorizeInference(testCase)
    [modelPath,configPath,bundle]=alternateDeployment(testCase);
    verifyDeploymentBundle(modelPath,configPath);
    verifyError(testCase,@() verifyRuntimeDeploymentSources(bundle),'NetrAI:RuntimeSourceMismatch');
    % The loader must fail before importing this deliberately synthetic model.
    verifyError(testCase,@() loadNetrAIModel(modelPath,configPath),'NetrAI:RuntimeSourceMismatch');
end

function testArtifactOrderingDoesNotChangeRuntimeRoles(testCase)
    manifest=testCase.TestData.manifest;
    manifest.artifacts=manifest.artifacts([5 3 1 4 2]);
    resolved=verifyRuntimeDeploymentSources(manifest);
    verifyEqual(testCase,string({resolved.role}),testCase.TestData.roles);
end

function testMissingDuplicateAndMalformedRolesFail(testCase)
    manifest=testCase.TestData.manifest;
    manifest.artifacts(5)=[];
    verifyError(testCase,@() verifyRuntimeDeploymentSources(manifest),'NetrAI:InvalidBundle');
    manifest=testCase.TestData.manifest;
    manifest.artifacts(5)=manifest.artifacts(1);
    verifyError(testCase,@() verifyRuntimeDeploymentSources(manifest),'NetrAI:InvalidBundle');
    manifest=testCase.TestData.manifest;
    manifest.artifacts(1).sha256='not-a-digest';
    verifyError(testCase,@() verifyRuntimeDeploymentSources(manifest),'NetrAI:InvalidBundle');
end

function [modelPath,configPath,bundle]=alternateDeployment(testCase)
    matlabDir=fullfile(testCase.TestData.root,'alternate','matlab');
    modelsDir=fullfile(matlabDir,'models'); mkdir(modelsDir);
    modelPath=fullfile(modelsDir,'synthetic.onnx');
    configPath=fullfile(modelsDir,'synthetic_config.json');
    writeFile(modelPath,'not-an-ONNX-model'); writeFile(configPath,'{"fixture":true}');
    paths=["synthetic.onnx","synthetic_config.json", ...
        "../src/m3_grading/preprocessModelInput.m", ...
        "../src/m3_grading/predictDRGrade.m", ...
        "../src/m4_explainability/calibrateConfidence.m", ...
        "../src/utils/assignScreeningTriage.m","../src/utils/readScreeningImage.m"];
    roles=["model","config",testCase.TestData.roles];
    artifacts=repmat(struct('role','','path','','sha256',''),7,1);
    for index=1:7
        target=fullfile(modelsDir,paths(index));
        if index>2
            folder=fileparts(target); if ~isfolder(folder), mkdir(folder); end
            copyfile(testCase.TestData.sources(index-2),target);
            if index==3, appendComment(target); end
        end
        artifacts(index)=struct('role',char(roles(index)),'path',char(paths(index)), ...
            'sha256',fileSHA256(target));
    end
    bundle=struct('schema_version',2,'artifacts',artifacts);
    writeFile(fullfile(modelsDir,'netrai_bundle.json'),jsonencode(bundle));
end

function appendComment(filePath)
    fid=fopen(filePath,'a'); assert(fid>=0);
    cleaner=onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid,'\n%% Synthetic alternate-deployment source mutation.\n');
end

function writeFile(filePath,contents)
    fid=fopen(filePath,'wb'); assert(fid>=0);
    cleaner=onCleanup(@() fclose(fid)); %#ok<NASGU>
    fwrite(fid,unicode2native(contents,'UTF-8'),'uint8');
end
