function tests=testValidationInputs
    tests=functiontests(localfunctions);
end

function setup(testCase)
    testCase.TestData.originalPath=path;
    base=fileparts(fileparts(mfilename('fullpath')));
    addpath(base,genpath(fullfile(base,'src')));
    root=tempname; mkdir(root); mkdir(fullfile(root,'train_images'));
    testCase.TestData.root=root;
    writetable(table(["0001";"0002"],[0;2],'VariableNames',{'id_code','diagnosis'}),fullfile(root,'train.csv'));
end

function teardown(testCase)
    path(testCase.TestData.originalPath);
    rmdir(testCase.TestData.root,'s');
end

function testMissingCohortPublishesUnresolvedAndPreservesLeadingZeros(testCase)
    root=testCase.TestData.root; output=fullfile(root,'validation');
    result=validatePipeline(root,'Mode','full','OutputDir',output);
    verifyEqual(testCase,result.requested,2);
    verifyEqual(testCase,result.successful,0);
    verifyEqual(testCase,result.errors,2);
    verifyEqual(testCase,result.coverage,0);
    verifyTrue(testCase,isnan(result.qwk));
    identity=jsondecode(fileread(fullfile(output,'run_identity.json')));
    verifyEqual(testCase,string({identity.expected_cohort.id_code}),["0001","0002"]);
    merged=aggregateValidationRuns(root,fullfile(root,'combined'));
    verifyEqual(testCase,merged.unresolved,2);
    verifyTrue(testCase,merged.expected_cohort_verified);
    verifyError(testCase,@() validatePipeline(root,'Mode','full','OutputDir',output),'NetrAI:OutputExists');
end

function testSelectionCannotChangeTruthLabels(testCase)
    root=testCase.TestData.root; selected=fullfile(root,'selected.csv');
    writetable(table("0001",4,'VariableNames',{'id_code','diagnosis'}),selected);
    verifyError(testCase,@() validatePipeline(root,'SelectionCsv',selected, ...
        'OutputDir',fullfile(root,'validation')),'NetrAI:InvalidSelection');
end

function testIdsCannotEscapeImageFolder(testCase)
    root=testCase.TestData.root;
    writetable(table("../image",0,'VariableNames',{'id_code','diagnosis'}),fullfile(root,'train.csv'));
    verifyError(testCase,@() validatePipeline(root,'OutputDir',fullfile(root,'validation')), ...
        'NetrAI:InvalidIDs');
end
