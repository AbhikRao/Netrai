function payload = verifyTelemedicineWorkflow(outputDir, fixtureName)
%VERIFYTELEMEDICINEWORKFLOW Seven deterministic conservation/identity fixtures.
% Fixture results do not establish population or clinical performance.
    arguments
        outputDir (1,1) string = ""
        fixtureName (1,1) string = "all"
    end
    fixtureDir = fileparts(mfilename('fullpath'));
    % Public entry point belongs in matlab/; preparation stays in release/.
    matlabDir = fixtureDir;
    if strlength(outputDir) == 0
        outputDir = fullfile(matlabDir, 'results', 'm5_endpoint3_fixtures_' + ...
            string(datetime('now', 'Format', 'yyyyMMdd_HHmmss')));
    end
    if isfolder(outputDir) || isfile(outputDir)
        error('NetrAI:M5:OutputExists', 'Fixture evidence requires a fresh output directory.');
    end
    mkdir(outputDir);
    modelPath = setupSimulinkModel(fullfile(outputDir, ...
        'NetrAI_Telemedicine_Model.slx'));
    load_system(modelPath);
    modelName = 'NetrAI_Telemedicine_Model';
    workspace = get_param(modelName, 'ModelWorkspace');
    assert(workspace.evalin('m5EndpointVersion') == 3);
    assert(numel(find_system(modelName, 'BlockType', 'EntityGenerator')) == 1);
    assert(isempty(find_system(modelName, 'BlockType', 'EntityReplicator')));
    close_system(modelName, 0);
    base = struct('totalPatients', 1, 'simulationDuration', 30, ...
        'interarrivalTime', 1000, 'queueCapacity', 4, 'acquisitionTime', 2, ...
        'recaptureTime', 4, 'uploadTime', 3, 'aiInferenceTime', 1, ...
        'reviewTimePerPatient', 20, 'clinicianCount', 1, ...
        'qualityRejectRate', 0, 'retryQualityRejectRate', 0, ...
        'maxRecaptureRetries', 2, 'recaptureAbandonRate', 0, ...
        'referRate', 0, 'uncertainRate', 0, 'autoClearRate', 1, ...
        'traceEnabled', true);
    % Each expected count is specified independently from model counters.
    cases = { ...
        'no_reject', struct(), [1 1 0 0 0 1 1 0]; ...
        'zero_retries', struct('qualityRejectRate', 1, 'maxRecaptureRetries', 0), ...
            [1 0 0 1 0 1 1 0]; ...
        'first_reject_then_success', struct('qualityRejectRate', 1), ...
            [1 1 0 0 0 2 2 1]; ...
        'repeated_rejection_exhaustion', struct('qualityRejectRate', 1, ...
            'retryQualityRejectRate', 1), [1 0 0 1 0 3 3 2]; ...
        'dropout', struct('qualityRejectRate', 1, 'recaptureAbandonRate', 1), ...
            [1 0 1 0 0 1 1 0]; ...
        'cutoff_mid_recapture', struct('qualityRejectRate', 1, ...
            'simulationDuration', 4), [1 0 0 0 1 1 1 0]; ...
        'clinician_congestion', struct('totalPatients', 20, ...
            'interarrivalTime', 3, 'acquisitionTime', 1, ...
            'uploadTime', 1, 'reviewTimePerPatient', 10, ...
            'referRate', 1, 'autoClearRate', 0), [11 2 0 0 9 11 10 0]};
    if fixtureName ~= "all"
        selected = string(cases(:, 1)) == fixtureName;
        if ~any(selected)
            error('NetrAI:M5:UnknownFixture', 'Unknown fixture %s.', fixtureName);
        end
        cases = cases(selected, :);
    end
    rows = cell(size(cases, 1), 1);
    for index = 1:size(cases, 1)
        parameters = base;
        overrides = cases{index, 2};
        names = fieldnames(overrides);
        for fieldIndex = 1:numel(names)
            parameters.(names{fieldIndex}) = overrides.(names{fieldIndex});
        end
        [row, simulation, ledger] = runTelemedicineScenario(modelPath, parameters);
        actual = [row.generated, row.completed, row.abandoned, ...
            row.quality_exhausted, row.in_system, ...
            row.acquisition_attempts_started, row.quality_checks_completed, ...
            row.recapture_returns];
        expected = cases{index, 3};
        assert(isequal(actual, expected), ...
            'NetrAI:M5:FixtureMismatch', '%s expected [%s], got [%s].', ...
            cases{index, 1}, num2str(expected), num2str(actual));
        if strcmp(cases{index, 1}, 'cutoff_mid_recapture')
            assert(row.recapture_pending == 1);
        elseif strcmp(cases{index, 1}, 'clinician_congestion')
            assert(row.reviewed == 2 && row.max_review_queue >= 7 && ...
                row.occupancy_at_stop.reviewN == 1);
            % Reviews start at 3, 13, 23 for arrivals at 3, 6, 9.
            % Their queue waits are 0, 7, 14; served patients each take 10 s.
            assert(abs(row.review_running_average_wait_seconds - 7) < 1e-9);
            assert(abs(row.review_service_running_average_time_seconds - 10) < 1e-9);
            % The reviewer is occupied continuously from time 3 through 30,
            % including the review unfinished at cutoff: 27/30 = 0.9.
            assert(abs(row.clinician_utilization - 0.9) < 1e-9);
        end
        row.fixture = cases{index, 1};
        row.expected_counts = expected;
        row.actual_counts = actual;
        row.passed = true;
        rows{index} = row;
        save(fullfile(outputDir, string(cases{index, 1}) + '.mat'), ...
            'simulation', 'ledger', 'row');
        fprintf('M5 v3 fixture %s: PASS\n', cases{index, 1});
    end
    payload = struct('schema_version', 1, 'endpoint_version', 3, ...
        'validation_scope', 'deterministic_engineering_fixtures', ...
        'fixture_count', numel(rows), 'all_passed', true, ...
        'results', {rows}, 'scope', ...
        'Deterministic engineered fixtures; not simulated population field performance.');
    payload.fixture_source_sha256 = fileSHA256([mfilename('fullpath') '.m']);
    payload.artifact_guard_checks = verifyArtifactGuards(modelPath, base, outputDir);
    fileId = fopen(fullfile(outputDir, 'fixture_results.json'), 'w');
    cleaner = onCleanup(@() fclose(fileId));
    fprintf(fileId, '%s\n', jsonencode(payload, 'PrettyPrint', true));
    clear cleaner;
end

function checks = verifyArtifactGuards(modelPath, parameters, outputDir)
% Modify copies only. No additional simulation is needed for these fail gates.
    guardsDir = fullfile(outputDir, 'artifact_guard_copies');
    mkdir(guardsDir);
    copyPath = fullfile(guardsDir, 'NetrAI_Telemedicine_Model.slx');
    copyfile(modelPath, copyPath);
    expectError(@() runTelemedicineScenario(copyPath, parameters), ...
        'NetrAI:M5:MissingWorkflowManifest');
    copyfile(modelPath + ".manifest.json", copyPath + ".manifest.json");
    fileId = fopen(copyPath, 'ab');
    fwrite(fileId, uint8(0), 'uint8');
    fclose(fileId);
    expectError(@() runTelemedicineScenario(copyPath, parameters), ...
        'NetrAI:M5:WorkflowIntegrity');
    copyfile(modelPath, copyPath, 'f');
    manifest = jsondecode(fileread(copyPath + ".manifest.json"));
    manifest.source_identity.source_sha256 = repmat('0', 1, 64);
    fileId = fopen(copyPath + ".manifest.json", 'w');
    fprintf(fileId, '%s\n', jsonencode(manifest, 'PrettyPrint', true));
    fclose(fileId);
    expectError(@() runTelemedicineScenario(copyPath, parameters), ...
        'NetrAI:M5:StaleWorkflow');
    checks = struct('missing_manifest_rejected', true, ...
        'changed_model_rejected', true, 'stale_source_identity_rejected', true);
end

function expectError(callable, identifier)
    try
        callable();
    catch exception
        assert(strcmp(exception.identifier, identifier), ...
            'NetrAI:M5:UnexpectedGuardError', ...
            'Expected %s, received %s.', identifier, exception.identifier);
        return;
    end
    error('NetrAI:M5:MissingGuardError', 'Expected fail gate %s was not raised.', identifier);
end
