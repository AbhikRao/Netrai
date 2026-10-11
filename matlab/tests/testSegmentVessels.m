function tests = testSegmentVessels
%TESTSEGMENTVESSELS Contrast-free inputs must not invent retinal vessels.
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    matlabDir = fileparts(fileparts(mfilename('fullpath')));
    testCase.TestData.originalPath = path;
    addpath(genpath(fullfile(matlabDir, 'src')));
end

function teardownOnce(testCase)
    path(testCase.TestData.originalPath);
end

function testBlackFrameHasNoVessels(testCase)
    verifyConstantFrame(testCase, uint8(0));
end

function testGreyFrameHasNoVessels(testCase)
    verifyConstantFrame(testCase, uint8(128));
end

function testWhiteFrameHasNoVessels(testCase)
    verifyConstantFrame(testCase, uint8(255));
end

function testEmptyFieldHasNoVessels(testCase)
    image = repmat(uint8(128), 128, 128);
    image(20:108, 62:66) = 20;
    vessels = segmentVessels(image, false(size(image)));
    testCase.verifyClass(vessels, 'logical');
    testCase.verifySize(vessels, size(image));
    testCase.verifyEqual(nnz(vessels), 0);
end

function testDarkLineStillProducesVesselPixels(testCase)
    image = repmat(uint8(128), 128, 128);
    line = false(size(image));
    line(20:108, 62:66) = true;
    image(line) = 20;
    field = true(size(image));
    vessels = segmentVessels(image, field);
    testCase.verifyClass(vessels, 'logical');
    testCase.verifySize(vessels, size(image));
    testCase.verifyGreaterThan(nnz(vessels & line), 20);
end

function testMismatchedFieldShapeIsRejected(testCase)
    image = repmat(uint8(128), 128, 128);
    testCase.verifyError(@() segmentVessels(image, true(127, 128)), ...
        'NetrAI:InvalidVesselMask');
end

function testNonfiniteGreenInputIsRejected(testCase)
    image = zeros(128, 128);
    image(64, 64) = NaN;
    testCase.verifyError(@() segmentVessels(image, true(size(image))), ...
        'NetrAI:InvalidVesselInput');
    image(64, 64) = Inf;
    testCase.verifyError(@() segmentVessels(image, true(size(image))), ...
        'NetrAI:InvalidVesselInput');
end

function testNonfiniteFieldIsRejected(testCase)
    image = repmat(uint8(128), 128, 128);
    field = ones(size(image));
    field(64, 64) = NaN;
    testCase.verifyError(@() segmentVessels(image, field), ...
        'NetrAI:InvalidVesselMask');
end

function testRgbInputIsRejected(testCase)
    testCase.verifyError(@() segmentVessels(zeros(128, 128, 3), true(128, 128)), ...
        'NetrAI:InvalidVesselInput');
end

function verifyConstantFrame(testCase, level)
    image = repmat(level, 128, 128);
    field = true(size(image));
    vessels = segmentVessels(image, field);
    testCase.verifyClass(vessels, 'logical');
    testCase.verifySize(vessels, size(image));
    testCase.verifyEqual(nnz(vessels), 0);
end
