function tests=testComputeMetrics
    tests=functiontests(localfunctions);
end

function setupOnce(testCase)
    testCase.TestData.originalPath=path;
    addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))),'src','utils'));
end

function teardownOnce(testCase)
    path(testCase.TestData.originalPath);
end

function testAllWrongClassHasZeroF1(testCase)
    metrics=computeMetrics([0 3;2 0]);
    verifyEqual(testCase,metrics.f1Score,[0 0]);
    verifyEqual(testCase,metrics.macroF1,0);
end

function testNoPredictedPositiveStillHasDefinedF1(testCase)
    metrics=computeMetrics([2 0;3 0]);
    verifyEqual(testCase,metrics.f1Score(2),0);
    verifyTrue(testCase,isnan(metrics.precision(2)));
end

function testEmptyCohortIsUndefinedNotAnError(testCase)
    metrics=computeMetrics(zeros(5));
    verifyTrue(testCase,isnan(metrics.accuracy));
    verifyTrue(testCase,all(isnan(metrics.f1Score)));
    verifyTrue(testCase,isnan(metrics.macroF1));
end

function testAbsentClassF1IsUndefined(testCase)
    metrics=computeMetrics(diag([2 3 0]));
    verifyEqual(testCase,metrics.f1Score(1:2),[1 1]);
    verifyTrue(testCase,isnan(metrics.f1Score(3)));
end

function testInvalidCountsAreRejected(testCase)
    for bad={[-1 0;0 1],[NaN 0;0 1],[Inf 0;0 1],ones(2,3)}
        verifyError(testCase,@() computeMetrics(bad{1}),'NetrAI:InvalidConfusionMatrix');
    end
end
