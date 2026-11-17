function report = runPayloadWhatIf(imageSizeMB, outputFile)
%RUNPAYLOADWHATIF Rerun the headline SimEvents scenarios with another upload size per image.
% The three headline scenarios of the published endpoint-v3 sweep (one reviewer, 10% assumed rejection, 0.25 / 1 / 5 Mbps: sweep
% scenarios 2, 8 and 14, same overrides and seeds) run once per size in imageSizeMB, with uploadTime = 2 x size x 8 / bandwidth.
% A 3.5 MB reference run is always added and compared with the published rows: a mismatch means this what-if no longer reproduces
% the sweep, and is reported. The published sweep, its files and the content-bound M5 sources are not changed. Planning what-if under
% the same unfitted assumptions, not field evidence.
%   report = runPayloadWhatIf([0.4 1.0], "payload_whatif.json")
    arguments
        imageSizeMB (1,:) double {mustBePositive, mustBeFinite}
        outputFile (1,1) string = ""
    end
    root = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(root, 'src')));
    modelPath = fullfile(root, 'models', 'NetrAI_Telemedicine_Model.slx');
    published = readtable(fullfile(root, '..', 'results', 'benchmarks', 'matlab_workflow_2026-09-28', 'simulink_scenario_sweep.csv'));
    headline = [2 0.25; 8 1.0; 14 5.0];                 % sweep scenario id, bandwidth (Mbps)
    patients = 100000; horizon = 250 * 8 * 3600;
    sizes = unique([3.5 imageSizeMB], 'stable');
    rows = cell(0, 1);
    for sizeMB = sizes
        for h = 1:height(headline)
            id = headline(h, 1); bandwidth = headline(h, 2);
            overrides = struct('totalPatients', patients, 'workingDays', 250, 'operatingHoursPerDay', 8, ...
                'simulationDuration', horizon, 'interarrivalTime', horizon / patients, 'queueCapacity', patients + 2, ...
                'imagesPerPatient', 2, 'imageSizeMB', sizeMB, 'bandwidthMbps', bandwidth, 'uploadTime', 2 * sizeMB * 8 / bandwidth, ...
                'clinicianCount', 1, 'qualityRejectRate', 0.10, 'retryQualityRejectRate', 0.10, ...
                'maxRecaptureRetries', 2, 'recaptureAbandonRate', 0, ...
                'qualityRandomSeed', 4200 + id, 'triageRandomSeed', 5200 + id, 'traceEnabled', false);
            detail = runTelemedicineScenario(modelPath, overrides);
            rows{end+1, 1} = struct('image_size_mb', sizeMB, 'bandwidth_mbps', bandwidth, 'sweep_scenario_id', id, ...
                'upload_seconds_per_patient', overrides.uploadTime, 'generated', detail.generated, ...
                'completed', detail.completed, 'auto_cleared', detail.auto_cleared, 'reviewed', detail.reviewed, ...
                'quality_exhausted', detail.quality_exhausted, 'abandoned', detail.abandoned, 'in_system', detail.in_system, ...
                'completion_rate', detail.completion_rate, 'max_upload_queue', detail.max_upload_queue, ...
                'model_sha256', string(detail.model_sha256), 'source_sha256', string(detail.source_identity.source_sha256)); %#ok<AGROW>
            fprintf('%.2f MB, %.2f Mbps: %d of %d completed, %d in system\n', sizeMB, bandwidth, ...
                detail.completed, detail.generated, detail.in_system);
        end
    end
    results = struct2table(vertcat(rows{:}));
    counts = ["generated" "completed" "auto_cleared" "reviewed" "quality_exhausted" "abandoned" "in_system"];
    reference = results(results.image_size_mb == 3.5, :);
    mismatches = strings(0, 1);
    if height(reference) ~= height(headline)
        mismatches(end+1, 1) = sprintf('%d reference rows, expected %d', height(reference), height(headline));
    end
    for h = 1:height(reference)
        expected = published(published.scenario_id == reference.sweep_scenario_id(h), :);
        if height(expected) ~= 1                      % a missing or repeated published row cannot pass silently
            mismatches(end+1, 1) = sprintf('scenario %d: %d published rows, expected 1', ...
                reference.sweep_scenario_id(h), height(expected)); %#ok<AGROW>
            continue
        end
        for name = counts
            if reference.(name)(h) ~= expected.(name)
                mismatches(end+1, 1) = sprintf('scenario %d %s: %d vs published %d', ...
                    reference.sweep_scenario_id(h), name, reference.(name)(h), expected.(name)); %#ok<AGROW>
            end
        end
    end
    report = struct('purpose', 'headline SimEvents scenarios rerun at other per-image upload sizes (planning what-if)', ...
        'scenarios', 'one reviewer, 10% assumed rejection, 0.25 / 1 / 5 Mbps (published sweep scenarios 2, 8, 14; same seeds)', ...
        'reference_matches_published', isempty(mismatches), 'reference_mismatches', {cellstr(mismatches)}, ...
        'results', table2struct(results), 'limitations', {{ ...
            'All other assumptions are those of the published sweep: unfitted quality rejection, routing and timing, zero dropout.', ...
            'Only the upload size changes; recompressing images to reach a smaller size changes some AI decisions (upload compression study).', ...
            'Working-hour-equivalent calendar; no shifts, outages or off-hours uploading.'}});
    if ~isempty(mismatches)
        warning('NetrAI:M5:WhatIfReference', 'The 3.5 MB reference does not reproduce the published sweep:\n%s', ...
            strjoin(mismatches, newline));
    end
    if strlength(outputFile) > 0
        fid = fopen(outputFile, 'w');
        fwrite(fid, jsonencode(report, 'PrettyPrint', true), 'char');
        fclose(fid);
    end
end
