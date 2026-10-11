function identity = telemedicineSourceIdentity()
%TELEMEDICINESOURCEIDENTITY Content-bind the M5 implementation and hash utilities.
% Normalized release paths remain stable when this reviewed source is relocated
% from its preparation folder to matlab/src/m5_simulink.
    m5Dir = fileparts(mfilename('fullpath'));
    required = sort(["defaultTelemedicineParameters.m", "recordTelemedicineEvent.m", ...
        "runSimulinkScenarioSweep.m", "runTelemedicineScenario.m", ...
        "setupSimulinkModel.m", "telemedicineSourceIdentity.m"]);
    entries = repmat(struct('path', '', 'sha256', ''), numel(required)+2, 1);
    builderDigest = "";
    for index = 1:numel(required)
        path = fullfile(m5Dir, required(index));
        if ~isfile(path)
            error('NetrAI:M5:MissingSource', 'Required workflow source %s is missing.', required(index));
        end
        digest = fileSHA256(path);
        entries(index) = struct('path', char("src/m5_simulink/" + required(index)), ...
            'sha256', char(digest));
        if required(index) == "setupSimulinkModel.m"
            builderDigest = string(digest);
        end
    end
    utilities = ["fileSHA256", "sha256Bytes"];
    for index = 1:numel(utilities)
        utilityPath = string(which(utilities(index)));
        [~, name, extension] = fileparts(utilityPath);
        if strlength(utilityPath) == 0 || ~isfile(utilityPath) || ...
                string(name) + string(extension) ~= utilities(index) + ".m"
            error('NetrAI:M5:MissingSource', 'Required hash utility %s is missing.', utilities(index));
        end
        entries(numel(required)+index) = struct( ...
            'path', char("src/utils/" + utilities(index) + ".m"), ...
            'sha256', char(fileSHA256(utilityPath)));
    end
    [~, order] = sort(string({entries.path}));
    entries = entries(order);
    digest = sha256Bytes(unicode2native(jsonencode(entries), 'UTF-8'));
    identity = struct('schema_version', 1, 'endpoint_version', 3, ...
        'builder_sha256', char(builderDigest), 'source_sha256', char(digest), ...
        'source_files', entries);
end
