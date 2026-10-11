function tests = testLocalizeOpticDiscFOV
    tests = functiontests(localfunctions);
end

function testDisconnectedBrighterDiscCannotOverrideSuppliedFov(testCase)
    [image, fov] = makeDiscImage(true);
    [center, radius] = localizeOpticDisc(image, image(:,:,2), fov);
    verifyTrue(testCase, fov(round(center(2)), round(center(1))));
    verifyLessThan(testCase, norm(center - [52 128]), 2);
    verifyGreaterThan(testCase, radius, 10);
    verifyLessThan(testCase, radius, 15);
end

function testExistingInsideDiscRemainsLocalized(testCase)
    [image, fov] = makeDiscImage(false);
    [center, radius] = localizeOpticDisc(image, image(:,:,2), fov);
    verifyLessThan(testCase, norm(center - [52 128]), 2);
    verifyGreaterThan(testCase, radius, 10);
    verifyLessThan(testCase, radius, 15);
end

function testEmptyFovDoesNotFabricateCoordinates(testCase)
    image = zeros(256, 256, 3, 'uint8');
    verifyError(testCase, @() localizeOpticDisc(image, image(:,:,2), false(256)), ...
        'NetrAI:EmptyFOV');
end

function testZeroImageWithClaimedFovHasNoDiscEvidence(testCase)
    image = zeros(256, 256, 3, 'uint8');
    verifyError(testCase, @() localizeOpticDisc(image, image(:,:,2), true(256)), ...
        'NetrAI:OpticDiscUnavailable');
end

function testInvalidFovShapeOrValueIsRejected(testCase)
    [image, ~] = makeDiscImage(false);
    verifyError(testCase, @() localizeOpticDisc(image, image(:,:,2), true(128)), ...
        'NetrAI:InvalidFOV');
    invalid = ones(256); invalid(1) = NaN;
    verifyError(testCase, @() localizeOpticDisc(image, image(:,:,2), invalid), ...
        'NetrAI:InvalidFOV');
end

function [image, fov] = makeDiscImage(addOutsideDisc)
    [xx, yy] = meshgrid(1:256, 1:256);
    fov = (xx-128).^2 + (yy-128).^2 <= 90^2;
    image = zeros(256, 256, 3, 'uint8');
    for channel = 1:3
        layer = image(:,:,channel); layer(fov) = 60; image(:,:,channel) = layer;
    end
    insideDisc = (xx-52).^2 + (yy-128).^2 <= 12^2;
    red = image(:,:,1); red(insideDisc) = 160;
    if addOutsideDisc
        outsideDisc = (xx-20).^2 + (yy-128).^2 <= 12^2;
        red(outsideDisc) = 255;
    end
    image(:,:,1) = red;
end
