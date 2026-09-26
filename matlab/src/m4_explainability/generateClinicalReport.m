% H1: generateClinicalReport - Generates a clinical DR screening report
%
% Syntax: reportPath = generateClinicalReport(img, segResults, grade, calProbs, checklist, heatmap, overlayImg, outputDir, gradcamStatus, gradcamQC, safety)
%
function reportPath = generateClinicalReport(img, segResults, grade, calProbs, checklist, heatmap, overlayImg, outputDir, gradcamStatus, gradcamQC, safety, modelDisplay)
    if nargin < 12 && ~isempty(heatmap)
        error('NetrAI:MissingModelDisplay','A classifier-space image is required for Grad-CAM.');
    end
    if nargin < 12, modelDisplay = img; end
    if nargin < 9
        gradcamStatus = "unavailable";
    end
    if nargin < 10
        gradcamQC = struct();
    end
    if nargin < 11
        safety = struct();
    end
    % Create output directory if not exists
    absOutDir = char(outputDir);
    if ~isfolder(absOutDir)
        mkdir(absOutDir);
    end
    reportPath = fullfile(absOutDir, 'ClinicalReport.pdf');
    reportPng = fullfile(absOutDir, 'ClinicalReport.png');
    
    % Create figure
    fig = figure('Visible', 'off', 'Position', [100, 100, 1400, 900]);
    
    % Layout
    t = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t, 'NetrAI Research Screening Report - Not a Diagnosis', 'FontSize', 20, 'FontWeight', 'bold');
    
    % Panel 1: Enhanced Fundus
    nexttile;
    imshow(img);
    title('Enhanced Fundus');
    
    % Panel 2: Grad-CAM. Never replace a failed explanation with synthetic data.
    nexttile;
    imshow(modelDisplay);
    if isempty(heatmap)
        title('Grad-CAM unavailable');
        text(0.5, 0.05, char(gradcamStatus), 'Units', 'normalized', ...
            'HorizontalAlignment', 'center', 'Color', 'white', ...
            'BackgroundColor', 'black', 'Interpreter', 'none', 'FontSize', 8);
    else
        hold on;
        h = imagesc(imresize(heatmap, size(modelDisplay, [1 2])));
        colormap(gca, jet);
        alpha(h, 0.4);
        axis image off;
        if isfield(gradcamQC,'shortcut_flag') && gradcamQC.shortcut_flag
            title('Referral-logit Grad-CAM - QC FLAG');
        else
            title('Referral-logit Grad-CAM');
        end
    end
    
    % Panel 3: Lesion overlay
    nexttile;
    imshow(overlayImg);
    title('Candidate Lesions (not pixel-mask validated)');
    
    % Panel 4: Text panel
    nexttile;
    axis off;
    str = sprintf('DR Grade: Level %d\n\n', grade);
    str = [str sprintf('Referable: %s\n\n', num2str(grade > 1))];
    str = [str sprintf('Post-hoc grade probability: %.2f%%\n\n', calProbs(grade+1)*100)];
    if isfield(safety, 'referableHeadProbability')
        str = [str sprintf('Referral head (shared backbone): %.2f%%\n', ...
            safety.referableHeadProbability*100)];
        str = [str sprintf('Head disagreement: %s\n', ...
            string(safety.headDisagreement))];
        str = [str sprintf('Triage: %s\n', string(safety.triageAction))];
    end
    text(0, 0.95, str, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
        'FontSize', 12, 'Interpreter', 'none');
    
    % Panel 5: Checklist
    nexttile;
    axis off;
    strChecklist = sprintf('Evidence Checklist:\n');
    for i=1:length(checklist)
        strChecklist = [strChecklist sprintf('%s: %s (%s)\n', checklist(i).criterion, checklist(i).status, checklist(i).detail)];
    end
    if isfield(safety, 'referableHeadProbability')
        agreement = "AGREE";
        if safety.headDisagreement, agreement = "DISAGREE - REVIEW"; end
        strChecklist = [strChecklist sprintf('Referral head: %s (%.1f%%)\n', ...
            agreement, safety.referableHeadProbability*100)];
    end
    text(0, 0.95, wrapReportText(strChecklist, 52), ...
        'Units', 'normalized', 'VerticalAlignment', 'top', ...
        'FontSize', 9, 'Interpreter', 'none');
    
    % Panel 6: Bar chart
    nexttile;
    b = bar(0:4, calProbs);
    title('DR Level Probabilities');
    xlabel('ICDR Level');
    ylabel('Probability');
    
    % Annotation
    annotation('textbox', [0, 0, 1, 0.05], 'String', sprintf('AI-assisted | Requires ophthalmologist validation | Generated: %s', datestr(now)), 'EdgeColor', 'none', 'HorizontalAlignment', 'center');
    
    % Save
    try
        exportgraphics(fig, reportPath, 'Resolution', 200);
        exportgraphics(fig, reportPng, 'Resolution', 200);
    catch
        saveas(fig, reportPng);
        try
            saveas(fig, reportPath);
        catch
            reportPath = reportPng;
        end
    end
    
    close(fig);
end

function wrapped = wrapReportText(value, maxCharacters)
    lines = splitlines(string(value));
    output = strings(0, 1);
    for i = 1:numel(lines)
        words = split(strtrim(lines(i)));
        current = "";
        for j = 1:numel(words)
            if strlength(current) > 0 && ...
                    strlength(current) + 1 + strlength(words(j)) > maxCharacters
                output(end + 1, 1) = current; %#ok<AGROW>
                current = words(j);
            elseif strlength(current) == 0
                current = words(j);
            else
                current = current + " " + words(j);
            end
        end
        output(end + 1, 1) = current; %#ok<AGROW>
    end
    wrapped = char(join(output, newline));
end
