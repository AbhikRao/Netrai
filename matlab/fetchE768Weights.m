function destination = fetchE768Weights(tag)
%FETCHE768WEIGHTS Download the E_768 research weights from the project's GitHub Release and verify their SHA-256.
%
% The expected hash is read from the committed deployment bundle (variant_R3_E_768/netrai_bundle.json); a file that does
% not match is deleted and never installed. Read the weights' terms in docs/MODEL_CARD.md before use.
    arguments
        tag (1,1) string = "e768-v1"
    end
    baseDir = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(baseDir, 'src')));
    variantDir = fullfile(baseDir, 'variant_R3_E_768');
    destination = fullfile(variantDir, 'netrai_fold0.onnx');
    bundle = jsondecode(fileread(fullfile(variantDir, 'netrai_bundle.json')));
    roles = string({bundle.artifacts.role});
    expected = string(bundle.artifacts(roles == "model").sha256);
    if isfile(destination) && strcmpi(string(fileSHA256(destination)), expected)
        fprintf('E_768 weights already present and verified: %s\n', destination);
        return
    end
    url = "https://github.com/AbhikRao/Netrai/releases/download/" + tag + "/netrai_E_768.onnx";
    partial = destination + ".part";
    cleanup = onCleanup(@() deleteIfExists(partial));
    fprintf('Downloading %s\n', url);
    websave(partial, url);
    if ~strcmpi(string(fileSHA256(partial)), expected)
        error('NetrAI:WeightsHashMismatch', 'Downloaded file does not match the SHA-256 in netrai_bundle.json; not installed.');
    end
    movefile(partial, destination);
    fprintf('Installed and verified: %s\n', destination);
end

function deleteIfExists(path)
    if isfile(path), delete(path); end
end
