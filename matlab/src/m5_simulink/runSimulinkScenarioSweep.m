function results = runSimulinkScenarioSweep(outputDir, totalPatients)
%RUNSIMULINKSCENARIOSWEEP Execute district-scale bandwidth/staffing scenarios.
%
% results = runSimulinkScenarioSweep()
% results = runSimulinkScenarioSweep(outputDir, totalPatients)

    arguments
        outputDir (1,1) string = ""
        totalPatients (1,1) double {mustBeInteger,mustBePositive} = 100000
    end

    matlabDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    if strlength(outputDir) == 0
        outputDir = fullfile(matlabDir, 'results', 'm5_scenario_sweep');
    end
    if ~isfolder(outputDir)
        mkdir(outputDir);
    end

    modelPath = fullfile(matlabDir, 'models', ...
        'NetrAI_Telemedicine_Model.slx');
    if ~isfile(modelPath)
        setupSimulinkModel(modelPath);
    end
    load_system(modelPath);
    modelName = "NetrAI_Telemedicine_Model";
    workspace = get_param(modelName, 'ModelWorkspace');

    bandwidths = [0.25 1.0 5.0];
    clinicians = [1 2];
    rejectRates = [0.05 0.10 0.20];
    scenarios = combinations(bandwidths, clinicians, rejectRates);
    scenarios.Properties.VariableNames = { ...
        'bandwidth_mbps', 'clinicians', 'quality_reject_rate'};

    rows = repmat(emptyResult(), height(scenarios), 1);
    annualSeconds = 250 * 8 * 3600;
    expectedArrivals = totalPatients + 1;
    for index = 1:height(scenarios)
        scenario = scenarios(index, :);
        rng(4200 + index, 'twister');
        workspace.assignin('totalPatients', totalPatients);
        workspace.assignin('simulationDuration', annualSeconds);
        workspace.assignin('interarrivalTime', annualSeconds / totalPatients);
        workspace.assignin('bandwidthMbps', scenario.bandwidth_mbps);
        workspace.assignin('uploadTime', 2 * 3.5 * 8 / scenario.bandwidth_mbps);
        workspace.assignin('clinicianCount', scenario.clinicians);
        workspace.assignin('qualityRejectRate', scenario.quality_reject_rate);

        started = tic;
        simulation = sim(modelName);
        elapsed = toc(started);
        row = emptyResult();
        row.scenario_id = index;
        row.total_patients_target = totalPatients;
        row.expected_arrivals = expectedArrivals;
        row.bandwidth_mbps = scenario.bandwidth_mbps;
        row.clinicians = scenario.clinicians;
        row.quality_reject_rate = scenario.quality_reject_rate;
        row.auto_cleared = round(finalValue(simulation, 'autoCleared'));
        row.reviewed = round(finalValue(simulation, 'reviewed'));
        row.recaptures = round(finalValue(simulation, 'recaptures'));
        row.completed = row.auto_cleared + row.reviewed;
        row.unfinished_at_stop = max(expectedArrivals - row.completed, 0);
        row.completion_rate = row.completed / expectedArrivals;
        row.review_rate = row.reviewed / max(row.completed, 1);
        row.recapture_rate_observed = row.recaptures / expectedArrivals;
        row.clinician_utilization = finalValue(simulation, 'clinicianUtilization');
        row.average_review_wait_seconds = finalValue( ...
            simulation, 'reviewQueueAverageWait');
        row.p90_running_average_review_wait_seconds = percentileValue( ...
            simulation, 'reviewQueueAverageWait', 90);
        row.max_acquisition_queue = maxValue(simulation, 'acquisitionQueueLength');
        row.max_upload_queue = maxValue(simulation, 'uploadQueueLength');
        row.max_ai_queue = maxValue(simulation, 'aiQueueLength');
        row.max_review_queue = maxValue(simulation, 'reviewQueueLength');
        row.throughput_patients_per_hour = row.completed / (250 * 8);
        row.required_review_clinicians_at_80pct_utilization = max(1, ceil( ...
            expectedArrivals * (1-scenario.quality_reject_rate) * (0.08+0.0123) ...
            * 30 / (annualSeconds * 0.80)));
        row.runtime_seconds = elapsed;
        rows(index) = row;
        fprintf(['M5 %02d/%02d: %.2f Mbps, %d clinician(s), reject %.0f%% ' ...
            '-> completion %.3f, max review queue %.0f\n'], ...
            index, height(scenarios), row.bandwidth_mbps, row.clinicians, ...
            100 * row.quality_reject_rate, row.completion_rate, ...
            row.max_review_queue);
    end

    results = struct2table(rows);
    writetable(results, fullfile(outputDir, 'simulink_scenario_sweep.csv'));
    payload = struct();
    payload.schema_version = 2;
    payload.model = 'NetrAI_Telemedicine_Model.slx';
    payload.scope = '100000+ annual district screening arrivals';
    payload.scenario_count = height(results);
    payload.results = table2struct(results);
    payload.limitations = { ...
        'Recapture requests remain unresolved; retry/return behavior is not yet simulated.', ...
        ['Routing uses configurable research assumptions pending EyeQ/external ' ...
         'validation.'], ...
        ['P90 is the percentile of the running average review-wait statistic; ' ...
         'the Python reference reports patient-level wait percentiles.']};
    json = jsonencode(payload, 'PrettyPrint', true);
    fileId = fopen(fullfile(outputDir, 'simulink_scenario_sweep.json'), 'w');
    cleaner = onCleanup(@() fclose(fileId));
    fprintf(fileId, '%s\n', json);
    clear cleaner;
    close_system(modelName, 0);
end


function row = emptyResult()
    row = struct( ...
        'endpoint_version', 2, ...
        'scenario_id', 0, ...
        'total_patients_target', 0, ...
        'expected_arrivals', 0, ...
        'bandwidth_mbps', 0, ...
        'clinicians', 0, ...
        'quality_reject_rate', 0, ...
        'auto_cleared', 0, ...
        'reviewed', 0, ...
        'recaptures', 0, ...
        'completed', 0, ...
        'unfinished_at_stop', 0, ...
        'completion_rate', 0, ...
        'review_rate', 0, ...
        'recapture_rate_observed', 0, ...
        'clinician_utilization', 0, ...
        'average_review_wait_seconds', 0, ...
        'p90_running_average_review_wait_seconds', 0, ...
        'max_acquisition_queue', 0, ...
        'max_upload_queue', 0, ...
        'max_ai_queue', 0, ...
        'max_review_queue', 0, ...
        'throughput_patients_per_hour', 0, ...
        'required_review_clinicians_at_80pct_utilization', 0, ...
        'runtime_seconds', 0);
end


function value = finalValue(simulation, name)
    series = simulation.get(name);
    if isempty(series) || isempty(series.Data)
        value = 0;
    else
        value = double(series.Data(end));
    end
end


function value = maxValue(simulation, name)
    series = simulation.get(name);
    if isempty(series) || isempty(series.Data)
        value = 0;
    else
        value = double(max(series.Data));
    end
end


function value = percentileValue(simulation, name, percentile)
    series = simulation.get(name);
    if isempty(series) || isempty(series.Data)
        value = 0;
    else
        value = double(prctile(series.Data, percentile));
    end
end
