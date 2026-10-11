function [row, simulation, ledger] = runTelemedicineScenario(modelPath, overrides)
%RUNTELEMEDICINESCENARIO Run one bound endpoint-v3 engineering scenario.
    arguments
        modelPath (1,1) string = ""
        overrides (1,1) struct = struct()
    end
    if strlength(modelPath) == 0
        m5Dir = fileparts(mfilename('fullpath'));
        modelPath = fullfile(fileparts(fileparts(m5Dir)), 'models', ...
            'NetrAI_Telemedicine_Model.slx');
    end
    params = defaultTelemedicineParameters();
    names = fieldnames(overrides);
    for index = 1:numel(names)
        if ~isfield(params, names{index})
            error('NetrAI:M5:UnknownParameter', 'Unknown parameter %s.', names{index});
        end
        params.(names{index}) = overrides.(names{index});
    end
    validateParameters(params);
    % There are no replacement patients or extra copies. Leave sufficient
    % storage for every scheduled arrival, including a possible cutoff arrival.
    minimumCapacity = floor(params.simulationDuration / params.interarrivalTime) + 2;
    params.queueCapacity = max(params.queueCapacity, minimumCapacity);
    [manifest, currentIdentity] = verifyWorkflowManifest(modelPath);
    modelName = "NetrAI_Telemedicine_Model";
    if bdIsLoaded(modelName)
        error('NetrAI:M5:ModelAlreadyLoaded', ...
            'Close the existing workflow model before running its saved, bound artifacts.');
    end
    load_system(modelPath);
    cleaner = onCleanup(@() closeWorkflowModel(modelName));
    workspace = get_param(modelName, 'ModelWorkspace');
    try
        endpointVersion = workspace.evalin('m5EndpointVersion');
        savedIdentity = workspace.evalin('m5SourceIdentity');
    catch
        error('NetrAI:M5:EndpointVersion', 'Saved workflow model provenance is missing.');
    end
    if ~isscalar(endpointVersion) || endpointVersion ~= 3
        error('NetrAI:M5:EndpointVersion', 'A content-bound endpoint-v3 model is required.');
    end
    if ~isstruct(savedIdentity) || ~isfield(savedIdentity, 'source_sha256') || ...
            string(savedIdentity.source_sha256) ~= string(currentIdentity.source_sha256)
        error('NetrAI:M5:StaleWorkflow', 'Saved model source identity does not match current workflow code.');
    end
    names = fieldnames(params);
    for index = 1:numel(names)
        workspace.assignin(names{index}, params.(names{index}));
    end
    recordTelemedicineEvent(0);
    started = tic;
    simulation = sim(modelName, 'SimulationMode', 'normal');
    runtime = toc(started);
    ledger = recordTelemedicineEvent(-1);
    [afterManifest, afterIdentity] = verifyWorkflowManifest(modelPath);
    if string(afterManifest.model_sha256) ~= string(manifest.model_sha256) || ...
            string(afterIdentity.source_sha256) ~= string(currentIdentity.source_sha256)
        error('NetrAI:M5:ArtifactChanged', 'Workflow artifacts changed during simulation.');
    end

    occupancySignals = {'generatorPending', 'acquisitionQueueN', ...
        'recaptureQueueN', 'uploadQueueN', 'aiQueueN', 'referralQueueN', ...
        'uncertainQueueN', 'reviewQueueN', 'acquisitionN', 'recaptureN', ...
        'uploadN', 'aiN', 'reviewN'};
    occupancy = struct();
    inSystem = 0;
    for index = 1:numel(occupancySignals)
        count = integerFinal(simulation, occupancySignals{index});
        occupancy.(occupancySignals{index}) = count;
        inSystem = inSystem + count;
    end
    row = struct('endpoint_version', 3, 'parameters', params, ...
        'model_sha256', manifest.model_sha256, ...
        'source_identity', currentIdentity, ...
        'generated', integerFinal(simulation, 'arrivalsDeparted') + ...
            occupancy.generatorPending, ...
        'auto_cleared', integerFinal(simulation, 'autoCleared'), ...
        'reviewed', integerFinal(simulation, 'reviewed'), ...
        'abandoned', integerFinal(simulation, 'abandoned'), ...
        'quality_exhausted', integerFinal(simulation, 'qualityExhausted'), ...
        'in_system', inSystem, ...
        'acquisition_attempts_started', integerFinal(simulation, 'acquisitionD') + ...
            occupancy.acquisitionN, ...
        'quality_checks_completed', integerFinal(simulation, 'acquisitionD'), ...
        'recapture_returns', integerFinal(simulation, 'recaptureD'), ...
        'recapture_pending', occupancy.recaptureQueueN + occupancy.recaptureN, ...
        'occupancy_at_stop', occupancy, ...
        'max_review_queue', integerMaximum(simulation, 'reviewQueueN'), ...
        'review_running_average_wait_seconds', finalValue(simulation, ...
            'reviewQueueRunningAverageWait'), ...
        'review_service_running_average_time_seconds', finalValue(simulation, ...
            'reviewServiceRunningAverageTime'), ...
        'clinician_utilization', occupancyUtilization(simulation, 'reviewN', ...
            params.simulationDuration, params.clinicianCount), ...
        'last_departure_clinician_utilization_statistic', finalValue( ...
            simulation, 'clinicianUtilization'), ...
        'runtime_seconds', runtime);
    row.completed = row.auto_cleared + row.reviewed;
    row.recap_requests = row.recapture_returns + row.recapture_pending;
    row.unfinished = row.abandoned + row.quality_exhausted + row.in_system;
    row.conservation_residual = row.generated - (row.completed + row.unfinished);
    if row.conservation_residual ~= 0
        error('NetrAI:M5:PatientConservation', ...
            'Patient conservation failed by %d entities.', row.conservation_residual);
    end
    if row.acquisition_attempts_started > row.generated * (params.maxRecaptureRetries + 1)
        error('NetrAI:M5:AttemptBound', 'Acquisition attempts exceeded the patient retry bound.');
    end
    row.completion_rate = ratio(row.completed, row.generated);
    row.abandonment_rate = ratio(row.abandoned, row.generated);
    row.quality_exhaustion_rate = ratio(row.quality_exhausted, row.generated);
    row.in_system_rate = ratio(row.in_system, row.generated);
    row.unfinished_rate = ratio(row.unfinished, row.generated);
    row.review_rate_among_completed = ratio(row.reviewed, row.completed);
    row.acquisition_attempts_per_generated_patient = ...
        ratio(row.acquisition_attempts_started, row.generated);
    row.max_acquisition_queue = integerMaximum(simulation, 'acquisitionQueueN');
    row.max_recapture_queue = integerMaximum(simulation, 'recaptureQueueN');
    row.max_upload_queue = integerMaximum(simulation, 'uploadQueueN');
    row.max_ai_queue = integerMaximum(simulation, 'aiQueueN');
    row.expected_quality_acceptance_probability = eventualAcceptance(params);
    reviewDemand = row.generated * row.expected_quality_acceptance_probability * ...
        (params.referRate + params.uncertainRate);
    row.required_review_clinicians_at_80pct_utilization = ceil( ...
        reviewDemand * params.reviewTimePerPatient / (params.simulationDuration * 0.8));
    row.limitations = { ...
        'Configured research assumptions, not patient field-performance estimates.', ...
        'Quality rejection and dropout are independent Bernoulli assumptions.', ...
        'Dropout is assessed when a permitted retry is requested, before return delay.', ...
        'Exhausted quality cases are unresolved terminal outcomes, not successful screening.', ...
        'Review waiting statistic is a running average; it is not a patient percentile.', ...
        'Annual horizon uses working-hour-equivalent seconds; real closed hours are not modeled.', ...
        'No shift calendar, review escalation or return travel distribution is implemented.'};
    if params.traceEnabled
        row.identity_checks = validateIdentity(ledger, row);
    else
        row.identity_checks = struct('trace_enabled', false);
    end
    clear cleaner;
end

function [manifest, identity] = verifyWorkflowManifest(modelPath)
    manifestPath = modelPath + ".manifest.json";
    if ~isfile(modelPath) || ~isfile(manifestPath)
        error('NetrAI:M5:MissingWorkflowManifest', ...
            'The workflow model requires its original content manifest.');
    end
    try
        manifest = jsondecode(fileread(manifestPath));
    catch
        error('NetrAI:M5:InvalidWorkflowManifest', 'Workflow manifest is not valid JSON.');
    end
    fields = {'schema_version', 'endpoint_version', 'model_file', ...
        'model_sha256', 'source_identity'};
    if ~isstruct(manifest) || ~isscalar(manifest) || ~all(isfield(manifest, fields)) || ...
            manifest.schema_version ~= 1 || manifest.endpoint_version ~= 3 || ...
            string(manifest.model_file) ~= "NetrAI_Telemedicine_Model.slx"
        error('NetrAI:M5:InvalidWorkflowManifest', 'Workflow manifest schema/version is invalid.');
    end
    if string(manifest.model_sha256) ~= string(fileSHA256(modelPath))
        error('NetrAI:M5:WorkflowIntegrity', 'The workflow SLX differs from its saved content manifest.');
    end
    identity = telemedicineSourceIdentity();
    if ~isstruct(manifest.source_identity) || ...
            ~isfield(manifest.source_identity, 'source_sha256') || ...
            ~isfield(manifest.source_identity, 'builder_sha256') || ...
            string(manifest.source_identity.source_sha256) ~= string(identity.source_sha256) || ...
            string(manifest.source_identity.builder_sha256) ~= string(identity.builder_sha256)
        error('NetrAI:M5:StaleWorkflow', ...
            'Workflow code changed after this SLX was generated; build a fresh model.');
    end
end

function checks = validateIdentity(ledger, row)
    generated = ledger(ledger(:, 1) == 1, 2);
    terminal = ledger(ismember(ledger(:, 1), 5:8), 2);
    if numel(generated) ~= row.generated || ...
            numel(unique(generated)) ~= row.generated
        error('NetrAI:M5:PatientIdentity', 'Generation IDs are missing or duplicated.');
    end
    if numel(unique(terminal)) ~= numel(terminal) || ...
            ~all(ismember(ledger(:, 2), generated))
        error('NetrAI:M5:PatientIdentity', 'A patient was copied, terminated twice or replaced.');
    end
    if numel(terminal) ~= row.completed + row.abandoned + row.quality_exhausted
        error('NetrAI:M5:PatientIdentity', 'Terminal identity records disagree with counters.');
    end
    entries = ledger(ledger(:, 1) == 2, :);
    for patientId = generated(:)'
        attempts = entries(entries(:, 2) == patientId, 3);
        if ~isequal(attempts(:)', 1:numel(attempts)) || ...
                numel(attempts) > row.parameters.maxRecaptureRetries + 1
            error('NetrAI:M5:AttemptIdentity', 'A retry did not preserve sequential patient attempts.');
        end
    end
    if height(entries) ~= row.acquisition_attempts_started || ...
            nnz(ledger(:, 1) == 4) ~= row.recapture_returns
        error('NetrAI:M5:AttemptIdentity', 'Attempt or return records disagree with counters.');
    end
    checks = struct('trace_enabled', true, 'unique_generation_ids', true, ...
        'terminal_ids_disjoint', true, 'same_patient_on_retry', true, ...
        'attempts_sequential_and_bounded', true);
end

function probability = eventualAcceptance(params)
    initialReject = params.qualityRejectRate;
    retryReject = params.retryQualityRejectRate;
    returnProbability = 1 - params.recaptureAbandonRate;
    probability = 1 - initialReject;
    reach = initialReject * returnProbability;
    for retry = 1:params.maxRecaptureRetries
        probability = probability + reach * (1 - retryReject);
        reach = reach * retryReject * returnProbability;
    end
end

function validateParameters(params)
    integers = {'totalPatients', 'workingDays', 'imagesPerPatient', ...
        'queueCapacity', 'clinicianCount', ...
        'qualityRandomSeed', 'triageRandomSeed'};
    for index = 1:numel(integers)
        validateattributes(params.(integers{index}), {'numeric'}, ...
            {'scalar', 'real', 'finite', 'integer', 'positive'}, '', integers{index});
    end
    validateattributes(params.maxRecaptureRetries, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'integer', 'nonnegative'});
    positive = {'operatingHoursPerDay', 'imageSizeMB', 'bandwidthMbps', ...
        'simulationDuration', 'interarrivalTime', 'acquisitionTime', ...
        'recaptureTime', 'uploadTime', 'aiInferenceTime', 'reviewTimePerPatient'};
    for index = 1:numel(positive)
        validateattributes(params.(positive{index}), {'numeric'}, ...
            {'scalar', 'real', 'finite', 'positive'}, '', positive{index});
    end
    probabilities = {'qualityRejectRate', 'retryQualityRejectRate', ...
        'recaptureAbandonRate', 'referRate', 'uncertainRate', 'autoClearRate'};
    for index = 1:numel(probabilities)
        validateattributes(params.(probabilities{index}), {'numeric'}, ...
            {'scalar', 'real', 'finite', '>=', 0, '<=', 1}, '', probabilities{index});
    end
    if abs(params.autoClearRate + params.referRate + params.uncertainRate - 1) > 1e-12
        error('NetrAI:M5:InvalidRouting', 'Routing probabilities must sum to one.');
    end
    validateattributes(params.traceEnabled, {'logical'}, {'scalar'});
end

function value = ratio(numerator, denominator)
    if denominator == 0, value = NaN; else, value = numerator / denominator; end
end

function utilization = occupancyUtilization(simulation, name, horizon, capacity)
% Integrate the piecewise-constant number of occupied servers through cutoff.
% The built-in utilization output updates on departures and can be stale when
% an entity is still being served at StopTime. Occupancy includes that service.
    series = simulation.get(name);
    if isempty(series) || isempty(series.Data)
        utilization = 0;
        return;
    end
    times = double(series.Time(:));
    counts = double(series.Data(:));
    if numel(times) ~= numel(counts) || any(~isfinite(times)) || ...
            any(diff(times) < 0) || any(~isfinite(counts)) || ...
            any(counts < 0 | counts > capacity)
        error('NetrAI:M5:InvalidOccupancySeries', 'Clinician occupancy statistics are invalid.');
    end
    selected = times <= horizon;
    times = times(selected);
    counts = counts(selected);
    if isempty(times)
        utilization = 0;
        return;
    end
    if times(1) > 0
        times = [0; times];
        counts = [0; counts];
    end
    durations = diff([times; horizon]);
    utilization = sum(durations .* counts) / (horizon * capacity);
end

function value = integerFinal(simulation, name)
    raw = finalValue(simulation, name);
    if ~isfinite(raw) || raw < 0 || abs(raw - round(raw)) > 1e-9
        error('NetrAI:M5:InvalidCounter', '%s is not a finite nonnegative integer.', name);
    end
    value = round(raw);
end

function value = integerMaximum(simulation, name)
    series = simulation.get(name);
    if isempty(series) || isempty(series.Data), value = 0;
    else, value = round(double(max(series.Data(:)))); end
end

function value = finalValue(simulation, name)
    series = simulation.get(name);
    if isempty(series) || isempty(series.Data), value = 0;
    else, value = double(series.Data(end)); end
end

function closeWorkflowModel(modelName)
    if bdIsLoaded(modelName), close_system(modelName, 0); end
end
