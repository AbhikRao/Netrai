function tests = testReadScreeningImage
    tests = functiontests(localfunctions);
end

function setup(testCase)
    folder = tempname;
    mkdir(folder);
    testCase.TestData.folder = folder;
end

function teardown(testCase)
    rmdir(testCase.TestData.folder,'s');
end

function testIndexedPaletteUsesColorValues(testCase)
    indices = uint8([0 1;2 3]);
    map = [1 0 0;0 1 0;0 0 1;1 1 0];
    imagePath = fullfile(testCase.TestData.folder,'palette.png');
    imwrite(indices,map,imagePath);
    expected = uint8(cat(3,[255 0;0 255],[0 255;0 255],[0 0;255 0]));
    verifyEqual(testCase,readScreeningImage(imagePath),expected);
end

function testRGBPixelsRemainExact(testCase)
    image = uint8(reshape(mod((0:107)*37,256),6,6,3));
    imagePath = fullfile(testCase.TestData.folder,'rgb.png');
    imwrite(image,imagePath);
    verifyEqual(testCase,readScreeningImage(imagePath),image);
end

function testJPEGAndTIFFUseDecodedColorBytes(testCase)
    image = uint8(reshape(mod((0:107)*37,256),6,6,3));
    for extension = ["jpg","tif"]
        imagePath = fullfile(testCase.TestData.folder,"color."+extension);
        imwrite(image,imagePath);
        verifyEqual(testCase,readScreeningImage(imagePath),imread(imagePath));
    end
end

function testGrayscaleUsesThreeEqualChannels(testCase)
    image = uint8(reshape(0:35,6,6));
    imagePath = fullfile(testCase.TestData.folder,'gray.png');
    imwrite(image,imagePath);
    verifyEqual(testCase,readScreeningImage(imagePath),repmat(image,1,1,3));
end

function testBilevelUsesBlackAndWhite(testCase)
    image = logical([0 1;1 0]);
    imagePath = fullfile(testCase.TestData.folder,'binary.png');
    imwrite(image,imagePath);
    verifyEqual(testCase,readScreeningImage(imagePath),repmat(uint8(image)*255,1,1,3));
end

function testTransparencyDoesNotChangeRGB(testCase)
    image = uint8(reshape(mod((0:107)*19,256),6,6,3));
    imagePath = fullfile(testCase.TestData.folder,'rgba.png');
    imwrite(image,imagePath,'Alpha',uint8(reshape(0:35,6,6)));
    verifyEqual(testCase,readScreeningImage(imagePath),image);
end

function testHighBitDepthIsExplicitlyRejected(testCase)
    imagePath = fullfile(testCase.TestData.folder,'sixteen.png');
    imwrite(uint16([0 65535;1024 4096]),imagePath);
    verifyError(testCase,@() readScreeningImage(imagePath),'NetrAI:InvalidImage');
end

function testBoundSnapshotUsesSamePaletteColors(testCase)
    imagePath = fullfile(testCase.TestData.folder,'palette-bound.png');
    imwrite(uint8([0 1;1 0]),[1 0 0;0 0 1],imagePath);
    [image,boundPath,cleanup] = readBoundValidationImage(imagePath,fileSHA256(imagePath));
    verifyEqual(testCase,image,readScreeningImage(imagePath));
    verifyEqual(testCase,fileSHA256(boundPath),fileSHA256(imagePath));
    clear cleanup
    verifyFalse(testCase,isfile(boundPath));
end
