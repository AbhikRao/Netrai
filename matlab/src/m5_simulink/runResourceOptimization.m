function results = runResourceOptimization()
%RUNRESOURCEOPTIMIZATION Evaluates screening scenarios over 250 working days
%
% Usage: results = runResourceOptimization()
%
% Runs a discrete event simulation for three scenarios:
% A: Manual Screening (No AI)
% B: Pure AI (Black Box, No Explainability)
% C: NetrAI (Human-in-the-Loop with CAFE)
% Generates publication-quality plots and comparison tables.

    % Simulation Parameters
    days = 250;
    patients_per_day = 400;
    images_per_day = patients_per_day * 2; % 800 images
    doctor_hours = 4;
    doctor_seconds = doctor_hours * 3600;

    % Initialize results struct
    results = struct();
    results.days = 1:days;

    % --- Scenario A: Manual Screening ---
    time_per_review_A = 180; % 3 min
    capacity_per_doc_A = floor(doctor_seconds / time_per_review_A); % 80
    
    docs_A = ceil(images_per_day / capacity_per_doc_A); % 10 docs
    
    backlog_A = zeros(1, days);
    available_docs_A = 2; % For backlog tracking, assume 2 available
    daily_capacity_A = available_docs_A * capacity_per_doc_A;
    cumulative_unreviewed_A = 0;
    
    for d = 1:days
        cumulative_unreviewed_A = cumulative_unreviewed_A + images_per_day - daily_capacity_A;
        backlog_A(d) = max(0, cumulative_unreviewed_A);
    end

    % --- Scenario B: Pure AI (No Explainability) ---
    distrust_factor = 0.50; % Docs verify 50%
    images_to_review_B = images_per_day * distrust_factor; % 400
    time_per_review_B = 90; % 90 sec
    capacity_per_doc_B = floor(doctor_seconds / time_per_review_B); % 160
    
    docs_B = ceil(images_to_review_B / capacity_per_doc_B); % 3 docs
    
    backlog_B = zeros(1, days);
    available_docs_B = 2;
    daily_capacity_B = available_docs_B * capacity_per_doc_B;
    cumulative_unreviewed_B = 0;
    
    for d = 1:days
        cumulative_unreviewed_B = cumulative_unreviewed_B + images_to_review_B - daily_capacity_B;
        backlog_B(d) = max(0, cumulative_unreviewed_B);
    end

    % --- Scenario C: NetrAI (Human-in-the-Loop) ---
    auto_clear_rate = 0.82;
    images_to_review_C = images_per_day * (1 - auto_clear_rate); % 144
    time_per_review_C = 30; % 30 sec
    capacity_per_doc_C = floor(doctor_seconds / time_per_review_C); % 480
    
    docs_C = ceil(images_to_review_C / capacity_per_doc_C); % 1 doc
    
    backlog_C = zeros(1, days);
    available_docs_C = 2;
    daily_capacity_C = available_docs_C * capacity_per_doc_C;
    cumulative_unreviewed_C = 0;
    
    for d = 1:days
        cumulative_unreviewed_C = cumulative_unreviewed_C + images_to_review_C - daily_capacity_C;
        backlog_C(d) = max(0, cumulative_unreviewed_C);
    end

    % Store results
    results.ScenA.backlog = backlog_A;
    results.ScenA.docs_needed = docs_A;
    results.ScenB.backlog = backlog_B;
    results.ScenB.docs_needed = docs_B;
    results.ScenC.backlog = backlog_C;
    results.ScenC.docs_needed = docs_C;
    
    % Costs and TAT
    doctor_cost_per_hr = 5000;
    cost_A = (docs_A * 4 * doctor_cost_per_hr) / patients_per_day;
    cost_B = (docs_B * 4 * doctor_cost_per_hr) / patients_per_day;
    cost_C = (docs_C * 4 * doctor_cost_per_hr) / patients_per_day;
    
    tat_A = (backlog_A(end) / images_per_day) + 1;
    tat_B = (backlog_B(end) / images_per_day) + 1;
    tat_C = 1; % No backlog

    % --- Visualization ---
    resDir = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'results');
    if ~exist(resDir, 'dir')
        mkdir(resDir);
    end
    
    f1 = figure('Name', 'Backlog Accumulation', 'Position', [100, 100, 800, 600], 'Visible', 'off');
    plot(1:days, backlog_A, 'r-', 'LineWidth', 2); hold on;
    plot(1:days, backlog_B, 'b-', 'LineWidth', 2);
    plot(1:days, backlog_C, 'g-', 'LineWidth', 2);
    set(gca, 'YScale', 'log');
    grid on;
    xlabel('Days');
    ylabel('Unreviewed Images (Log Scale)');
    title('Backlog Accumulation over 250 Days (Assuming 2 Doctors Available)');
    legend('Manual', 'Pure AI', 'NetrAI', 'Location', 'northwest');
    saveas(f1, fullfile(resDir, 'backlog_accumulation.png'));
    
    f2 = figure('Name', 'Doctors Needed', 'Position', [150, 150, 600, 500], 'Visible', 'off');
    bar([docs_A, docs_B, docs_C], 'FaceColor', [0.2 0.6 0.8]);
    set(gca, 'XTickLabel', {'Manual', 'Pure AI', 'NetrAI'});
    ylabel('Ophthalmologists Needed');
    title('Doctors Needed for Daily Caseload (800 images)');
    grid on;
    saveas(f2, fullfile(resDir, 'doctors_needed.png'));
    
    f3 = figure('Name', 'Turnaround Time', 'Position', [200, 200, 600, 500], 'Visible', 'off');
    bar([tat_A, tat_B, tat_C], 'FaceColor', [0.8 0.4 0.2]);
    set(gca, 'XTickLabel', {'Manual', 'Pure AI', 'NetrAI'});
    ylabel('Average Turnaround Time (Days)');
    title('Turnaround Time per Screening');
    grid on;
    saveas(f3, fullfile(resDir, 'turnaround_time.png'));
    
    f4 = figure('Name', 'Cost Per Screening', 'Position', [250, 250, 600, 500], 'Visible', 'off');
    bar([cost_A, cost_B, cost_C], 'FaceColor', [0.2 0.8 0.4]);
    set(gca, 'XTickLabel', {'Manual', 'Pure AI', 'NetrAI'});
    ylabel('Cost per Screening (INR)');
    title('Estimated Cost per Screening');
    grid on;
    saveas(f4, fullfile(resDir, 'cost_per_screening.png'));

    % --- Console Output ---
    fprintf('\n=== Resource Optimization Summary ===\n');
    fprintf('%-15s | %-12s | %-10s | %-10s\n', 'Scenario', 'Docs Needed', 'TAT (Days)', 'Cost (INR)');
    fprintf(repmat('-', 1, 55));
    fprintf('\n');
    fprintf('%-15s | %-12d | %-10.1f | %-10.0f\n', 'Manual', docs_A, tat_A, cost_A);
    fprintf('%-15s | %-12d | %-10.1f | %-10.0f\n', 'Pure AI', docs_B, tat_B, cost_B);
    fprintf('%-15s | %-12d | %-10.1f | %-10.0f\n', 'NetrAI', docs_C, tat_C, cost_C);
    fprintf(repmat('-', 1, 55));
    fprintf('\n\n');
end
