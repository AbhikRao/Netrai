function modelPath = setupSimulinkModel(outputPath)
%SETUPSIMULINKMODEL Build the executable NetrAI SimEvents district workflow.
%
% modelPath = setupSimulinkModel()
% modelPath = setupSimulinkModel('/path/to/NetrAI_Telemedicine_Model.slx')
%
% One patient is one SimEvents entity. The model includes acquisition,
% quality-driven recapture, bandwidth upload, AI processing, four-way
% screening triage, shared clinician review, and terminal counts.

    arguments
        outputPath (1,1) string = ""
    end

    matlabDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    if strlength(outputPath) == 0
        outputPath = fullfile(matlabDir, 'models', ...
            'NetrAI_Telemedicine_Model.slx');
    end
    outputPath = string(outputPath);
    outputDir = fileparts(outputPath);
    if ~isfolder(outputDir)
        mkdir(outputDir);
    end

    modelName = "NetrAI_Telemedicine_Model";
    if bdIsLoaded(modelName)
        close_system(modelName, 0);
    end
    if isfile(outputPath)
        delete(outputPath);
    end

    load_system('simulink');
    load_system('sldelib');
    new_system(modelName, 'Model');

    params = defaultParameters();
    workspace = get_param(modelName, 'ModelWorkspace');
    names = fieldnames(params);
    for index = 1:numel(names)
        workspace.assignin(names{index}, params.(names{index}));
    end

    set_param(modelName, ...
        'StopTime', 'simulationDuration', ...
        'SolverType', 'Variable-step', ...
        'Solver', 'VariableStepDiscrete', ...
        'SaveTime', 'on', ...
        'ReturnWorkspaceOutputs', 'on', ...
        'Description', ['NetrAI district telemedicine digital twin: ' ...
        '100,000 patients/year, recapture, bandwidth, AI, and review queues.']);

    add_block('sldelib/Entity Generator', modelName + "/Patient Arrivals", ...
        'Position', [30 135 145 195], ...
        'GenerationMethod', 'Time-based', ...
        'TimeSource', 'Dialog', ...
        'Period', 'interarrivalTime', ...
        'GenerateEntityAtSimulationStart', 'on', ...
        'EntityType', 'Structured', ...
        'AttributeName', 'Attribute1', ...
        'AttributeInitialValue', '2');

    addQueue(modelName, "Acquisition Queue", [190 140 285 200], ...
        'acquisitionQueueCapacity');
    addServer(modelName, "Image Acquisition", [330 140 445 200], ...
        'acquisitionTime', qualityAction());

    add_block('sldelib/Entity Output Switch', modelName + "/Quality Gate", ...
        'Position', [490 120 580 220], ...
        'NumberOutputPorts', '2', ...
        'SwitchingCriterion', 'From attribute', ...
        'SwitchAttributeName', 'Attribute1');

    addQueue(modelName, "Recapture Queue", [625 35 720 95], ...
        'recaptureQueueCapacity');
    addServer(modelName, "Recapture Delay", [765 35 880 95], ...
        'recaptureTime', '');
    add_block('sldelib/Entity Terminator', modelName + "/Recapture Required", ...
        'Position', [925 40 1035 90], ...
        'NumberEntitiesArrived', 'on');

    addQueue(modelName, "Upload Queue", [625 225 720 285], ...
        'uploadQueueCapacity');
    addServer(modelName, "Bandwidth Upload", [765 225 880 285], ...
        'uploadTime', '');
    addQueue(modelName, "AI Queue", [925 225 1020 285], ...
        'aiQueueCapacity');
    addServer(modelName, "AI Inference", [1065 225 1180 285], ...
        'aiInferenceTime', triageAction());

    add_block('sldelib/Entity Output Switch', modelName + "/Safety Triage", ...
        'Position', [1225 190 1315 315], ...
        'NumberOutputPorts', '3', ...
        'SwitchingCriterion', 'From attribute', ...
        'SwitchAttributeName', 'Attribute1');

    add_block('sldelib/Entity Terminator', modelName + "/Auto Clear", ...
        'Position', [1375 145 1465 195], ...
        'NumberEntitiesArrived', 'on');
    addQueue(modelName, "Referral Queue", [1370 230 1465 290], ...
        'reviewQueueCapacity');
    addQueue(modelName, "Uncertain Queue", [1370 345 1465 405], ...
        'reviewQueueCapacity');
    add_block('sldelib/Entity Input Switch', modelName + "/Review Merge", ...
        'Position', [1515 250 1595 385], ...
        'NumberInputPorts', '2', ...
        'ActivePortSelection', 'All', ...
        'SwitchingCriterion', 'Round robin');
    addQueue(modelName, "Review Queue", [1640 290 1735 350], ...
        'reviewQueueCapacity');
    addServer(modelName, "Clinician Review", [1780 290 1895 350], ...
        'reviewTimePerPatient', '');
    set_param(modelName + "/Clinician Review", ...
        'Capacity', 'clinicianCount', ...
        'Utilization', 'on', ...
        'AverageWait', 'on');
    add_block('sldelib/Entity Terminator', modelName + "/Reviewed", ...
        'Position', [1945 295 2035 345], ...
        'NumberEntitiesArrived', 'on');

    connectEntity(modelName, "Patient Arrivals", 1, "Acquisition Queue", 1);
    connectEntity(modelName, "Acquisition Queue", 1, "Image Acquisition", 1);
    connectEntity(modelName, "Image Acquisition", 1, "Quality Gate", 1);
    connectEntity(modelName, "Quality Gate", 1, "Recapture Queue", 1);
    connectEntity(modelName, "Recapture Queue", 1, "Recapture Delay", 1);
    connectEntity(modelName, "Recapture Delay", 1, "Recapture Required", 1);
    connectEntity(modelName, "Quality Gate", 2, "Upload Queue", 1);
    connectEntity(modelName, "Upload Queue", 1, "Bandwidth Upload", 1);
    connectEntity(modelName, "Bandwidth Upload", 1, "AI Queue", 1);
    connectEntity(modelName, "AI Queue", 1, "AI Inference", 1);
    connectEntity(modelName, "AI Inference", 1, "Safety Triage", 1);
    connectEntity(modelName, "Safety Triage", 1, "Auto Clear", 1);
    connectEntity(modelName, "Safety Triage", 2, "Referral Queue", 1);
    connectEntity(modelName, "Safety Triage", 3, "Uncertain Queue", 1);
    connectEntity(modelName, "Referral Queue", 1, "Review Merge", 1);
    connectEntity(modelName, "Uncertain Queue", 1, "Review Merge", 2);
    connectEntity(modelName, "Review Merge", 1, "Review Queue", 1);
    connectEntity(modelName, "Review Queue", 1, "Clinician Review", 1);
    connectEntity(modelName, "Clinician Review", 1, "Reviewed", 1);

    addStatisticSink(modelName, "Auto Clear", "autoCleared", [1525 115 1635 145]);
    addStatisticSink(modelName, "Reviewed", "reviewed", [2075 270 2185 300]);
    addStatisticSink(modelName, "Recapture Required", "recaptures", [1075 45 1185 75]);
    addStatisticSink(modelName, "Clinician Review", ...
        "clinicianUtilization", [1925 375 2060 405], 2);
    addStatisticSink(modelName, "Clinician Review", ...
        "reviewWait", [1925 420 2060 450], 1);
    addStatisticSink(modelName, "Acquisition Queue", ...
        "acquisitionQueueLength", [315 75 455 105], 1);
    addStatisticSink(modelName, "Upload Queue", ...
        "uploadQueueLength", [625 305 755 335], 1);
    addStatisticSink(modelName, "AI Queue", ...
        "aiQueueLength", [925 305 1055 335], 1);
    addStatisticSink(modelName, "Review Queue", ...
        "reviewQueueLength", [1640 390 1770 420], 1);
    addStatisticSink(modelName, "Review Queue", ...
        "reviewQueueAverageWait", [1640 435 1795 465], 2);

    set_param(modelName, 'ZoomFactor', 'FitSystem');
    save_system(modelName, outputPath);
    close_system(modelName, 0);
    close_system('sldelib', 0);

    modelPath = outputPath;
    fprintf('Created executable SimEvents model: %s\n', modelPath);
    fprintf('Base scenario: %d patients over %d working days.\n', ...
        params.totalPatients, params.workingDays);
end


function params = defaultParameters()
    params = struct();
    params.totalPatients = 100000;
    params.workingDays = 250;
    params.operatingHoursPerDay = 8;
    params.simulationDuration = params.workingDays * ...
        params.operatingHoursPerDay * 3600;
    params.interarrivalTime = params.simulationDuration / params.totalPatients;
    params.imagesPerPatient = 2;
    params.imageSizeMB = 3.5;
    params.bandwidthMbps = 1.0;
    params.uploadTime = params.imagesPerPatient * params.imageSizeMB * 8 / ...
        params.bandwidthMbps;
    params.acquisitionTime = 45;
    params.recaptureTime = 120;
    params.aiInferenceTime = 3.5;
    params.reviewTimePerPatient = 30;
    params.clinicianCount = 1;
    params.qualityRejectRate = 0.10;
    params.referRate = 0.08;
    params.uncertainRate = 0.0123;
    params.autoClearRate = 1 - params.referRate - params.uncertainRate;
    % Do not cap annual demand at an arbitrary 10,000 entities.  That older
    % limit propagated upload congestion back to acquisition and silently
    % prevented tens of thousands of scheduled patients from entering the
    % low-bandwidth scenarios.  The queues now retain the complete annual
    % cohort so the measured backlog is visible rather than discarded.
    params.acquisitionQueueCapacity = params.totalPatients + 1;
    params.recaptureQueueCapacity = params.totalPatients + 1;
    params.uploadQueueCapacity = params.totalPatients + 1;
    params.aiQueueCapacity = params.totalPatients + 1;
    params.reviewQueueCapacity = params.totalPatients + 1;
end


function addQueue(modelName, blockName, position, capacity)
    add_block('sldelib/Entity Queue', modelName + "/" + blockName, ...
        'Position', position, ...
        'Capacity', capacity, ...
        'QueueType', 'FIFO', ...
        'NumberEntitiesInBlock', 'on', ...
        'AverageWait', 'on');
end


function addServer(modelName, blockName, position, serviceTime, action)
    add_block('sldelib/Entity Server', modelName + "/" + blockName, ...
        'Position', position, ...
        'Capacity', '1', ...
        'ServiceTimeSource', 'Dialog', ...
        'ServiceTimeValue', serviceTime);
    if strlength(string(action)) > 0
        set_param(modelName + "/" + blockName, ...
            'ServiceCompleteAction', action);
    end
end


function action = qualityAction()
    action = [ ...
        'if rand() < qualityRejectRate' newline ...
        '    entity.Attribute1 = 1;' newline ...
        'else' newline ...
        '    entity.Attribute1 = 2;' newline ...
        'end'];
end


function action = triageAction()
    action = [ ...
        'u = rand();' newline ...
        'if u < autoClearRate' newline ...
        '    entity.Attribute1 = 1;' newline ...
        'elseif u < autoClearRate + referRate' newline ...
        '    entity.Attribute1 = 2;' newline ...
        'else' newline ...
        '    entity.Attribute1 = 3;' newline ...
        'end'];
end


function addStatisticSink(modelName, sourceName, variableName, position, signalIndex)
    if nargin < 5
        signalIndex = 1;
    end
    sinkName = "Log " + variableName;
    add_block('simulink/Sinks/To Workspace', modelName + "/" + sinkName, ...
        'Position', position, ...
        'VariableName', variableName, ...
        'SaveFormat', 'Timeseries');

    sourcePorts = get_param(modelName + "/" + sourceName, 'PortHandles');
    sinkPorts = get_param(modelName + "/" + sinkName, 'PortHandles');
    if numel(sourcePorts.Outport) < signalIndex
        error('NetrAI:M5:MissingStatisticPort', ...
            '%s does not expose requested statistic port %d.', ...
            sourceName, signalIndex);
    end
    add_line(modelName, sourcePorts.Outport(signalIndex), sinkPorts.Inport(1), ...
        'autorouting', 'on');
end


function connectEntity(modelName, sourceName, sourceIndex, destinationName, destinationIndex)
%CONNECTENTITY Connect right-side entity ports, ignoring top statistic ports.
    sourcePorts = get_param(modelName + "/" + sourceName, 'PortHandles');
    destinationPorts = get_param(modelName + "/" + destinationName, 'PortHandles');

    sourceHandles = sourcePorts.Outport;
    sourcePositions = zeros(numel(sourceHandles), 2);
    for index = 1:numel(sourceHandles)
        sourcePositions(index, :) = get_param(sourceHandles(index), 'Position');
    end
    rightEdge = max(sourcePositions(:, 1));
    entitySources = sourceHandles(sourcePositions(:, 1) == rightEdge);
    entitySourcePositions = sourcePositions(sourcePositions(:, 1) == rightEdge, :);
    [~, sourceOrder] = sort(entitySourcePositions(:, 2));
    entitySources = entitySources(sourceOrder);

    destinationHandles = destinationPorts.Inport;
    destinationPositions = zeros(numel(destinationHandles), 2);
    for index = 1:numel(destinationHandles)
        destinationPositions(index, :) = get_param(destinationHandles(index), 'Position');
    end
    [~, destinationOrder] = sort(destinationPositions(:, 2));
    entityDestinations = destinationHandles(destinationOrder);

    add_line(modelName, entitySources(sourceIndex), ...
        entityDestinations(destinationIndex), 'autorouting', 'on');
end
