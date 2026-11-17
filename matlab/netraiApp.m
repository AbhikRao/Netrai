function app = netraiApp(options)
%NETRAIAPP Screening console for live demonstrations of the NetrAI prototype.
%
%   netraiApp                     opens the console
%   netraiApp(Image="fundus.png") opens it with an image already chosen
%   app = netraiApp(...)          also returns handles: app.Figure, app.screen(path), app.setView(name), app.showCapacity()
%
% One window for the whole demonstration: choose a fundus photograph, run the same main_pipeline the command line uses, and read
% the routing decision, the model's attention map, the candidate lesion marks and the district capacity scenarios. Nothing is
% computed here that main_pipeline does not return; a failed step shows its error, never a substitute result.
% Research prototype: outputs are not a diagnosis. Options: Visible ("on"/"off"), Pipeline (a function handle with main_pipeline's
% interface; tests pass a stub), Image (a path).

    arguments
        options.Visible (1,1) string {mustBeMember(options.Visible, ["on" "off"])} = "on"
        options.Pipeline (1,1) function_handle = @main_pipeline
        options.Image (1,1) string = ""
    end

    baseDir = fileparts(mfilename('fullpath'));
    projectDir = fileparts(baseDir);
    addpath(genpath(fullfile(baseDir, 'src')));
    C = palette();
    gradeNames = ["No DR" "Mild NPDR" "Moderate NPDR" "Severe NPDR" "Proliferative DR"];

    lesionModel = findLesionModel(baseDir, projectDir);       % [] unless the research lesion segmenter is installed locally

    state = struct('imagePath', "", 'sitePolicy', "", 'result', [], 'view', "photo", 'photo', [], 'loaded', strings(1, 0));

    % ---------------------------------------------------------------- window
    fig = uifigure('Name', 'NetrAI screening console (research prototype)', 'Position', [60 50 1340 830], ...
        'Color', C.paper, 'Visible', 'off');
    if isprop(fig, 'Theme'), fig.Theme = 'light'; end       % readable on a projector whatever the desktop theme
    root = uigridlayout(fig, [3 3], 'RowHeight', {58, '1x', 26}, 'ColumnWidth', {292, '1x', 372}, ...
        'Padding', [20 12 20 14], 'RowSpacing', 12, 'ColumnSpacing', 14, 'BackgroundColor', C.paper);

    % ---------------------------------------------------------------- header
    header = uigridlayout(root, [2 2], 'RowHeight', {'1x', 20}, 'ColumnWidth', {'1x', 330}, 'Padding', [2 0 2 0], ...
        'RowSpacing', 0, 'BackgroundColor', C.paper);
    header.Layout.Row = 1; header.Layout.Column = [1 3];
    place(uilabel(header, 'Text', 'NetrAI', 'FontSize', 28, 'FontWeight', 'bold', 'FontColor', C.ink), 1, 1);
    place(uilabel(header, 'Text', 'Explainable diabetic-retinopathy screening for rural clinics', 'FontSize', 13, ...
        'FontColor', C.muted), 2, 1);
    pill = uilabel(header, 'Text', 'RESEARCH PROTOTYPE  ·  NOT A DIAGNOSIS', 'FontSize', 11, 'FontWeight', 'bold', ...
        'FontColor', C.paper, 'BackgroundColor', C.ink, 'HorizontalAlignment', 'center');
    place(pill, 1, 2);
    place(uilabel(header, 'Text', 'A clinician reviews every referral', 'FontSize', 11, 'FontColor', C.muted, ...
        'HorizontalAlignment', 'right'), 2, 2);

    % ---------------------------------------------------------------- left: inputs
    left = uipanel(root, 'BackgroundColor', C.card, 'BorderType', 'none');
    left.Layout.Row = 2; left.Layout.Column = 1;
    lg = uigridlayout(left, [13 1], 'RowHeight', {20, 38, 34, 14, 20, 30, 30, 44, 22, 22, 50, '1x', 78}, ...
        'Padding', [16 16 16 14], 'RowSpacing', 8, 'BackgroundColor', C.card);
    place(sectionLabel(lg, '1   FUNDUS PHOTOGRAPH'), 1, 1);
    chooseBtn = uibutton(lg, 'Text', 'Choose image…', 'FontSize', 13, 'ButtonPushedFcn', @(~, ~) chooseImage(), ...
        'Tooltip', 'A colour fundus photograph, as the camera saved it');
    place(chooseBtn, 2, 1);
    fileLabel = uilabel(lg, 'Text', 'No image chosen', 'FontColor', C.muted, 'FontSize', 12, 'WordWrap', 'on', ...
        'VerticalAlignment', 'top');
    place(fileLabel, 3, 1);
    place(sectionLabel(lg, '2   MODEL AND SITE'), 5, 1);
    variantDrop = uidropdown(lg, 'Items', {'Automatic (E_768 if installed)', 'E_768 (768 px, two models)', 'Legacy (512 px)'}, ...
        'ItemsData', {'auto', 'E_768', 'legacy'}, 'Value', 'auto', 'FontSize', 12);
    place(variantDrop, 6, 1);
    policyRow = uigridlayout(lg, [1 2], 'ColumnWidth', {'1x', 64}, 'Padding', [0 0 0 0], 'ColumnSpacing', 6, 'BackgroundColor', C.card);
    place(policyRow, 7, 1);
    uibutton(policyRow, 'Text', 'Site calibration file…', 'FontSize', 12, 'ButtonPushedFcn', @(~, ~) choosePolicy(), ...
        'Tooltip', 'A site policy written by calibrateSite (E_768 only)');
    uibutton(policyRow, 'Text', 'Clear', 'FontSize', 12, 'ButtonPushedFcn', @(~, ~) clearPolicy());
    policyLabel = uilabel(lg, 'Text', 'No site calibration: the cut-off is the frozen one and reports say UNCALIBRATED.', ...
        'FontColor', C.muted, 'FontSize', 11, 'WordWrap', 'on', 'VerticalAlignment', 'top');
    place(policyLabel, 8, 1);
    reportBox = uicheckbox(lg, 'Text', 'Also write the PDF report', 'Value', true, 'FontSize', 12);
    place(reportBox, 9, 1);
    lesionBox = uicheckbox(lg, 'Text', 'Learned lesion candidates (adds about 40 s)', 'Value', false, 'FontSize', 12, ...
        'Visible', onOff(~isempty(lesionModel)), 'Tooltip', 'Research lesion segmenter; adds a fourth picture');
    place(lesionBox, 10, 1);
    runBtn = uibutton(lg, 'Text', 'Run screening', 'FontSize', 16, 'FontWeight', 'bold', 'BackgroundColor', C.accent, ...
        'FontColor', [1 1 1], 'Enable', 'off', 'ButtonPushedFcn', @(~, ~) runScreening());
    place(runBtn, 11, 1);
    place(uilabel(lg, 'Text', ['Images are processed on this computer only. Lesion marks are unverified candidates and the ' ...
        'attention map shows model evidence, not a lesion outline.'], 'FontColor', C.muted, 'FontSize', 11, 'WordWrap', 'on', ...
        'VerticalAlignment', 'bottom'), 13, 1);

    % ---------------------------------------------------------------- centre: image views and capacity
    tabs = uitabgroup(root);
    tabs.Layout.Row = 2; tabs.Layout.Column = 2;
    screenTab = uitab(tabs, 'Title', '  Screening  ', 'BackgroundColor', C.card);
    sg = uigridlayout(screenTab, [3 1], 'RowHeight', {34, '1x', 50}, 'Padding', [14 12 14 10], 'RowSpacing', 8, 'BackgroundColor', C.card);
    viewRow = uigridlayout(sg, [1 5], 'ColumnWidth', {132, 132, 132, 132, '1x'}, 'Padding', [0 0 0 0], 'ColumnSpacing', 6, 'BackgroundColor', C.card);
    viewBtn.photo = uibutton(viewRow, 'Text', 'Photograph', 'ButtonPushedFcn', @(~, ~) setView("photo"));
    viewBtn.attention = uibutton(viewRow, 'Text', 'Model attention', 'ButtonPushedFcn', @(~, ~) setView("attention"));
    viewBtn.lesions = uibutton(viewRow, 'Text', 'Lesion candidates', 'ButtonPushedFcn', @(~, ~) setView("lesions"));
    viewBtn.learned = uibutton(viewRow, 'Text', 'Learned lesions', 'ButtonPushedFcn', @(~, ~) setView("learned"), ...
        'Visible', onOff(~isempty(lesionModel)));
    imageAxes = uiaxes(sg, 'BackgroundColor', C.card, 'Color', C.well, 'XTick', [], 'YTick', [], 'Box', 'off');
    imageAxes.XAxis.Visible = 'off'; imageAxes.YAxis.Visible = 'off';
    imageAxes.Toolbar.Visible = 'off'; disableDefaultInteractivity(imageAxes);
    place(imageAxes, 2, 1);
    viewCaption = uilabel(sg, 'Text', 'Choose a fundus photograph to begin.', 'FontColor', C.muted, 'FontSize', 12, 'WordWrap', 'on', ...
        'VerticalAlignment', 'top');
    place(viewCaption, 3, 1);

    capTab = uitab(tabs, 'Title', '  District capacity (SimEvents)  ', 'BackgroundColor', C.card);
    cg = uigridlayout(capTab, [4 1], 'RowHeight', {34, '1x', 118, 48}, 'Padding', [14 12 14 10], 'RowSpacing', 8, 'BackgroundColor', C.card);
    capRow = uigridlayout(cg, [1 6], 'ColumnWidth', {62, 56, 150, 66, '1x', 176}, 'Padding', [0 0 0 0], 'ColumnSpacing', 6, 'BackgroundColor', C.card);
    uilabel(capRow, 'Text', 'Reviewers', 'FontSize', 12, 'FontColor', C.muted);
    reviewersDrop = uidropdown(capRow, 'Items', {'1', '2'}, 'ItemsData', {1, 2}, 'Value', 1, 'ValueChangedFcn', @(~, ~) showCapacity());
    uilabel(capRow, 'Text', 'Images sent for recapture', 'FontSize', 12, 'FontColor', C.muted, 'HorizontalAlignment', 'right');
    rejectDrop = uidropdown(capRow, 'Items', {'5%', '10%', '20%'}, 'ItemsData', {0.05, 0.10, 0.20}, 'Value', 0.10, ...
        'ValueChangedFcn', @(~, ~) showCapacity());
    uilabel(capRow, 'Text', '');
    uibutton(capRow, 'Text', 'Open the SimEvents model', 'ButtonPushedFcn', @(~, ~) openCapacityModel());
    capAxes = uiaxes(cg, 'BackgroundColor', C.card);
    capAxes.Toolbar.Visible = 'off'; disableDefaultInteractivity(capAxes);
    place(capAxes, 2, 1);
    capTable = uitable(cg, 'ColumnName', {'Link, Mbps', 'Completed', 'In system', 'Out of retakes', 'Upload queue peak', 'Done, %'}, ...
        'RowName', {}, 'FontSize', 11, 'ColumnWidth', {78, '1x', '1x', '1x', 124, 70});
    place(capTable, 3, 1);
    capCaption = uilabel(cg, 'Text', '', 'FontColor', C.muted, 'FontSize', 11, 'WordWrap', 'on', 'VerticalAlignment', 'top');
    place(capCaption, 4, 1);

    % ---------------------------------------------------------------- right: the result
    right = uipanel(root, 'BackgroundColor', C.card, 'BorderType', 'none');
    right.Layout.Row = 2; right.Layout.Column = 3;
    rg = uigridlayout(right, [6 1], 'RowHeight', {84, 56, 176, 92, '1x', 36}, 'Padding', [16 16 16 14], 'RowSpacing', 10, 'BackgroundColor', C.card);
    banner = uilabel(rg, 'Text', 'NO RESULT YET', 'FontSize', 26, 'FontWeight', 'bold', 'FontColor', C.muted, ...
        'BackgroundColor', C.well, 'HorizontalAlignment', 'center');
    place(banner, 1, 1);
    bannerNote = uilabel(rg, 'Text', 'The routing decision appears here after a run.', 'FontSize', 13, 'FontColor', C.ink, ...
        'WordWrap', 'on', 'VerticalAlignment', 'top');
    place(bannerNote, 2, 1);
    dg = uigridlayout(rg, [7 2], 'RowHeight', repmat({'1x'}, 1, 7), 'ColumnWidth', {136, '1x'}, 'Padding', [0 0 0 0], ...
        'RowSpacing', 2, 'ColumnSpacing', 8, 'BackgroundColor', C.card);
    place(dg, 3, 1);
    detailNames = ["Screening level" "Most probable grade" "Referral output" "Image quality score" "Model" "Referral cut-off" "Time for this image"];
    detail = gobjects(1, numel(detailNames));
    for i = 1:numel(detailNames)
        uilabel(dg, 'Text', detailNames(i), 'FontSize', 12, 'FontColor', C.muted);
        detail(i) = uilabel(dg, 'Text', '–', 'FontSize', 12, 'FontWeight', 'bold', 'FontColor', C.ink, 'WordWrap', 'on');
    end
    notices = uilabel(rg, 'Text', '', 'FontSize', 12, 'FontColor', C.ink, 'WordWrap', 'on', 'VerticalAlignment', 'top', ...
        'Interpreter', 'html');
    place(notices, 4, 1);
    probAxes = uiaxes(rg, 'BackgroundColor', C.card);
    probAxes.Toolbar.Visible = 'off'; disableDefaultInteractivity(probAxes);
    place(probAxes, 5, 1);
    reportBtn = uibutton(rg, 'Text', 'Open the PDF report', 'Enable', 'off', 'ButtonPushedFcn', @(~, ~) openReport());
    place(reportBtn, 6, 1);

    % ---------------------------------------------------------------- footer
    statusLabel = uilabel(root, 'Text', 'Ready.', 'FontSize', 11, 'FontColor', C.muted);
    statusLabel.Layout.Row = 3; statusLabel.Layout.Column = [1 3];

    styleViewButtons();
    drawProbabilities([]);
    showCapacity();
    if strlength(options.Image) > 0
        setImage(options.Image);
    end
    fig.Visible = options.Visible;

    app = struct('Figure', fig, 'screen', @(path) screenImage(path, false), 'setImage', @setImage, 'setView', @setView, ...
        'showCapacity', @showCapacity, 'state', @currentState, 'Banner', banner, 'BannerNote', bannerNote, 'Details', detail, 'Notices', notices, ...
        'CapacityTable', capTable, 'RunButton', runBtn, 'ReportButton', reportBtn, 'Tabs', tabs, 'Status', statusLabel, ...
        'VariantDropDown', variantDrop, 'LesionBox', lesionBox, 'ReportBox', reportBox, 'ReviewersDropDown', reviewersDrop, 'RejectDropDown', rejectDrop);
    if nargout == 0, clear app; end

    % ================================================================ actions
    function chooseImage()
        [name, folder] = uigetfile({'*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp', 'Fundus photographs'}, 'Choose a fundus photograph');
        figure(fig);
        if isequal(name, 0), return; end
        setImage(string(fullfile(folder, name)));
    end

    function setImage(path)
        path = string(path);
        try
            photo = readScreeningImage(path);
        catch ME
            uialert(fig, ME.message, 'This file could not be read as an image');
            return;
        end
        state.imagePath = path; state.photo = photo; state.result = [];
        [~, name, ext] = fileparts(path);
        fileLabel.Text = name + ext + sprintf('  (%d × %d pixels)', size(photo, 2), size(photo, 1));
        fileLabel.FontColor = C.ink;
        runBtn.Enable = 'on';
        resetResult();
        setView("photo");
        statusLabel.Text = "Image chosen. Press Run screening.";
    end

    function choosePolicy()
        [name, folder] = uigetfile({'*.json', 'Site policy (JSON)'}, 'Choose the site calibration file');
        figure(fig);
        if isequal(name, 0), return; end
        state.sitePolicy = string(fullfile(folder, name));
        policyLabel.Text = "Site calibration: " + string(name) + " (checked against the model when the screening runs).";
        policyLabel.FontColor = C.ink;
    end

    function clearPolicy()
        state.sitePolicy = "";
        policyLabel.Text = 'No site calibration: the cut-off is the frozen one and reports say UNCALIBRATED.';
        policyLabel.FontColor = C.muted;
    end

    function runScreening()
        if strlength(state.imagePath) == 0, return; end
        screenImage(state.imagePath, true);
    end

    function s = currentState()
        s = state;
    end

    function result = screenImage(path, fromButton)
        % Run the pipeline on one image and show the result. A failure is shown in the window; a programmatic caller
        % (app.screen) also gets the error, so a script cannot mistake a failed run for a result.
        result = [];
        if string(path) ~= state.imagePath || isempty(state.photo)
            setImage(path);
            if string(path) ~= state.imagePath, return; end
        end
        variant = string(variantDrop.Value);
        firstUse = ~ismember(variant, state.loaded);
        message = "Quality check, grading, attention map and retinal analysis…";
        if firstUse
            message = "Loading the model. The first run in a MATLAB session takes up to two minutes; later images take seconds.";
        end
        progress = [];
        if fig.Visible == "on"
            progress = uiprogressdlg(fig, 'Title', 'Screening', 'Message', message, 'Indeterminate', 'on');
        end
        closeProgress = onCleanup(@() closeDialog(progress)); %#ok<NASGU>
        runBtn.Enable = 'off'; statusLabel.Text = "Running…"; drawnow;
        demoRoot = fullfile(baseDir, 'results', 'demo');
        if ~isfolder(demoRoot), mkdir(demoRoot); end
        started = tic;
        try
            callArgs = {'OutputDir', tempname(demoRoot), 'Verbose', false, 'GenerateReport', logical(reportBox.Value), ...
                'RunGradCAM', true, 'EnforceQuality', true, 'Variant', variant};
            if ~isempty(lesionModel) && lesionBox.Value
                callArgs = [callArgs, {'LearnedLesions', true, 'LesionOnnxPath', lesionModel.onnx, ...
                    'LesionThresholdsPath', lesionModel.thresholds}];
            end
            if strlength(state.sitePolicy) > 0
                callArgs = [callArgs, {'SitePolicy', state.sitePolicy}];
            end
            result = options.Pipeline(state.imagePath, callArgs{:});
        catch ME
            runBtn.Enable = 'on';
            statusLabel.Text = "The run failed: " + string(ME.message);
            if fig.Visible == "on"
                closeDialog(progress);
                uialert(fig, ME.message, 'The screening did not finish');
            end
            if ~fromButton, rethrow(ME); end
            return;
        end
        seconds = toc(started);
        state.loaded = union(state.loaded, variant);
        state.result = result;
        runBtn.Enable = 'on';
        showResult(result, seconds, firstUse);
    end

    function showResult(r, seconds, firstUse)
        timeText = sprintf('%.0f s', seconds);
        if firstUse, timeText = timeText + " (includes loading the model)"; end
        lines = strings(1, 0);
        if string(r.status) == "rejected"
            setBanner('RECAPTURE', C.recapture);
            bannerNote.Text = strjoin(string(r.qualityFeedback), ' ');
            values = ["not graded", "not graded", "not graded", sprintf('%.2f (accepted from 0.30)', r.iqs), ...
                variantName(r), "–", timeText];
            if isfield(r, 'rejectionReason') && string(r.rejectionReason) == "not_fundus"
                lines(end + 1) = notice(C.refer, "This does not look like a fundus photograph.");
            else
                lines(end + 1) = notice(C.recapture, "The image quality is too low to grade. Nothing was predicted.");
            end
            drawProbabilities([]);
            reportBtn.Enable = 'off';
        else
            switch string(r.triageAction)
                case "refer"
                    setBanner('REFER', C.refer);
                    bannerNote.Text = 'Referable diabetic retinopathy predicted. Send to a clinician for review.';
                case "uncertain_human_review"
                    setBanner('HUMAN REVIEW', C.review);
                    bannerNote.Text = 'The grade and the referral output disagree, so a person decides.';
                case "auto_clear"
                    setBanner('NO REFERRAL', C.clear);
                    bannerNote.Text = 'No referable diabetic retinopathy predicted. This is not a clearance; routine re-screening applies.';
                otherwise
                    setBanner(upper(char(r.triageAction)), C.recapture);
                    bannerNote.Text = '';
            end
            probs = double(r.probabilities(:))';
            [pBest, best] = max(probs);
            status = string(r.calibration.calibration_status);
            values = [sprintf('%d of 4 (refer from 2)', r.grade), sprintf('%s (%.0f%%)', gradeNames(best), pBest * 100), ...
                sprintf('%.0f%%', r.referableHeadProbability * 100), sprintf('%.2f (accepted from 0.30)', r.iqs), ...
                variantName(r), shortStatus(status), timeText];
            if isfield(r, 'inputWarning') && strlength(string(r.inputWarning)) > 0
                lines(end + 1) = notice(C.refer, "This does not look like a fundus photograph; the result is not meaningful.");
            end
            if startsWith(status, "UNCALIBRATED")
                lines(end + 1) = notice(C.review, "Cut-off not calibrated for this camera. Calibrate with labelled images of the site before screening.");
            end
            if contains(string(r.gradcamStatus), "QC FLAG")
                lines(end + 1) = notice(C.review, "Attention map flagged: evidence near the image border. Read it with care.");
            elseif ~startsWith(string(r.gradcamStatus), "available")
                lines(end + 1) = notice(C.muted, "Attention map: " + string(r.gradcamStatus) + ".");
            end
            drawProbabilities(probs, r.grade);
            reportBtn.Enable = onOff(isfield(r, 'reportPath') && strlength(string(r.reportPath)) > 0);
        end
        for k = 1:numel(detail), detail(k).Text = values(k); end
        if isempty(lines)
            lines = string(notice(C.muted, "No warnings for this image."));
        end
        notices.Text = strjoin(lines, '<br>');
        if string(r.status) == "rejected", setView("photo"); else, setView("attention"); end
        statusLabel.Text = "Done in " + timeText + ". Research output, not a diagnosis.";
        drawnow;
    end

    function resetResult()
        setBanner('NO RESULT YET', C.well); banner.FontColor = C.muted;
        bannerNote.Text = 'The routing decision appears here after a run.';
        for k = 1:numel(detail), detail(k).Text = '–'; end
        notices.Text = '';
        drawProbabilities([]);
        reportBtn.Enable = 'off';
    end

    function setBanner(text, colour)
        banner.Text = text; banner.BackgroundColor = colour; banner.FontColor = [1 1 1];
    end

    function setView(name)
        name = string(name);
        r = state.result;
        haveVisuals = ~isempty(r) && isfield(r, 'visuals');
        if name ~= "photo" && ~haveVisuals, name = "photo"; end
        if name == "learned" && ~haveLearned(r), name = "lesions"; end
        state.view = name;
        styleViewButtons();
        cla(imageAxes);
        switch name
            case "attention"
                if isempty(r.visuals.heatmap)
                    showImage(r.visuals.modelDisplay);
                    viewCaption.Text = "No attention map for this image (" + string(r.gradcamStatus) + "). Showing the image the model graded.";
                else
                    showImage(blendHeat(r.visuals.modelDisplay, r.visuals.heatmap));
                    viewCaption.Text = ['Grad-CAM of the referral output on the image the model graded: warm areas carried the evidence. ' ...
                        'It shows where the model looked, not the outline of a lesion.'];
                end
            case "learned"
                showImage(imread(r.learnedLesions.overlayPath));
                viewCaption.Text = "Research lesion segmenter: " + learnedCounts(r.learnedLesions) + ". Candidates, not confirmed " + ...
                    "lesions (Dice on a locked test: 0.60 / 0.43 / 0.26 for the first three); they do not change the grade.";
            case "lesions"
                showImage(r.visuals.overlayImage);
                viewCaption.Text = ['Candidate marks from the image-processing detectors (optic disc, fovea, vessels, possible lesions). ' ...
                    'They are unverified and do not change the grade.'];
            otherwise
                if isempty(state.photo)
                    viewCaption.Text = 'Choose a fundus photograph to begin.';
                else
                    showImage(state.photo);
                    viewCaption.Text = 'The photograph as read from the file. The model always grades this original image.';
                end
        end
    end

    function showImage(img)
        image(imageAxes, img);
        axis(imageAxes, 'image'); imageAxes.YDir = 'reverse';
        imageAxes.XAxis.Visible = 'off'; imageAxes.YAxis.Visible = 'off';
    end

    function styleViewButtons()
        names = string(fieldnames(viewBtn))';
        haveVisuals = ~isempty(state.result) && isfield(state.result, 'visuals');
        for n = names
            b = viewBtn.(n);
            selected = n == state.view;
            b.BackgroundColor = ternary(selected, C.ink, C.well);
            b.FontColor = ternary(selected, [1 1 1], C.ink);
            b.FontWeight = ternary(selected, 'bold', 'normal');
            b.Enable = onOff(n == "photo" || (haveVisuals && (n ~= "learned" || haveLearned(state.result))));
        end
    end

    function drawProbabilities(probs, grade)
        cla(probAxes);
        if isempty(probs)
            bar(probAxes, 0:4, zeros(1, 5), 'FaceColor', C.well, 'EdgeColor', 'none');
        else
            b = bar(probAxes, 0:4, probs, 0.72, 'FaceColor', 'flat', 'EdgeColor', 'none');
            b.CData = repmat(C.barMuted, 5, 1);
            b.CData(grade + 1, :) = ternary(grade >= 2, C.refer, C.clear);
            text(probAxes, 0:4, probs + 0.04, compose('%.0f%%', probs * 100), 'HorizontalAlignment', 'center', ...
                'FontSize', 10, 'Color', C.ink);
        end
        probAxes.XTick = 0:4; probAxes.XTickLabel = {'No DR', 'Mild', 'Moderate', 'Severe', 'PDR'};
        probAxes.YLim = [0 1.12]; probAxes.YTick = []; probAxes.YAxis.Visible = 'off';
        probAxes.Box = 'off'; probAxes.TickLength = [0 0]; probAxes.FontSize = 10; probAxes.Color = C.card;
        probAxes.XAxis.Color = C.muted;
        title(probAxes, 'Probability of each grade', 'FontSize', 11, 'FontWeight', 'normal', 'Color', C.muted);
    end

    function openReport()
        r = state.result;
        if isempty(r) || ~isfield(r, 'reportPath') || ~isfile(r.reportPath)
            uialert(fig, 'No report file for this image.', 'Report'); return;
        end
        try
            open(char(r.reportPath));
        catch ME
            uialert(fig, "The report is at " + string(r.reportPath) + " (" + string(ME.message) + ")", 'Report');
        end
    end

    function showCapacity()
        csvPath = fullfile(projectDir, 'results', 'benchmarks', 'matlab_workflow_2026-09-28', 'simulink_scenario_sweep.csv');
        cla(capAxes);
        if ~isfile(csvPath)
            capCaption.Text = "The recorded scenario table is not in this checkout: " + string(csvPath);
            capTable.Data = {};
            return;
        end
        sweep = readtable(csvPath);
        rows = sweep(sweep.clinicians == reviewersDrop.Value & abs(sweep.quality_reject_rate - rejectDrop.Value) < 1e-9, :);
        rows = sortrows(rows, 'bandwidth_mbps');
        completion = 100 * rows.completion_rate;
        b = bar(capAxes, 1:height(rows), completion, 0.55, 'FaceColor', 'flat', 'EdgeColor', 'none');
        b.CData = repmat(C.accent, height(rows), 1);
        b.CData(completion < 90, :) = repmat(C.refer, nnz(completion < 90), 1);
        text(capAxes, 1:height(rows), completion + 4, compose('%.2f%%', completion), 'HorizontalAlignment', 'center', ...
            'FontSize', 13, 'FontWeight', 'bold', 'Color', C.ink);
        capAxes.XTick = 1:height(rows); capAxes.XTickLabel = compose('%g Mbps link', rows.bandwidth_mbps);
        capAxes.YLim = [0 112]; capAxes.YTick = [0 50 100]; capAxes.YGrid = 'on'; capAxes.Box = 'off';
        capAxes.FontSize = 12; capAxes.Color = C.card; capAxes.TickLength = [0 0];
        ylabel(capAxes, 'Patients completed in a year (%)');
        title(capAxes, sprintf('100,000 patients a year, %d %s, %.0f%% of images sent for recapture', reviewersDrop.Value, ...
            ternary(reviewersDrop.Value == 1, 'reviewer', 'reviewers'), 100 * rejectDrop.Value), 'FontSize', 13, 'FontWeight', 'normal', 'Color', C.ink);
        capTable.Data = [compose('%g', rows.bandwidth_mbps), num2cell(rows.completed), num2cell(rows.in_system), ...
            num2cell(rows.quality_exhausted), num2cell(rows.max_upload_queue), compose('%.2f', completion)];
        capCaption.Text = ['Recorded SimEvents runs (28 September 2026), not field data: two 3.5 MB images per patient, a patient every ' ...
            '72 s, 30 s per review, at most two recaptures, nobody drops out. On a 0.25 Mbps link each upload takes longer than the gap ' ...
            'between patients, so a queue builds that a second reviewer cannot clear.'];
    end

    function openCapacityModel()
        try
            open_system(fullfile(baseDir, 'models', 'NetrAI_Telemedicine_Model.slx'));
        catch ME
            uialert(fig, ME.message, 'The SimEvents model could not be opened');
        end
    end

    % ================================================================ small helpers
    function label = sectionLabel(parent, text)
        label = uilabel(parent, 'Text', text, 'FontSize', 11, 'FontWeight', 'bold', 'FontColor', C.accent);
    end

    function text = notice(colour, message)
        text = sprintf('<span style="color:rgb(%d,%d,%d);font-weight:bold;">&#9679;</span> %s', round(colour * 255), message);
    end

    function name = variantName(r)
        name = "–";
        if isfield(r, 'model_variant')
            name = string(r.model_variant);
            if name == "E_768", name = "E_768 (768 px, two models)"; elseif name == "legacy", name = "Legacy (512 px)"; end
        end
    end
end

% ==================================================================== file-local helpers
function C = palette()
    C = struct('paper', [0.953 0.945 0.925], 'card', [1 1 1], 'well', [0.925 0.929 0.925], 'ink', [0.086 0.118 0.157], ...
        'muted', [0.376 0.412 0.455], 'accent', [0.047 0.345 0.380], 'refer', [0.690 0.149 0.118], 'review', [0.698 0.416 0.020], ...
        'clear', [0.118 0.420 0.306], 'recapture', [0.243 0.329 0.478], 'barMuted', [0.776 0.800 0.816]);
end

function place(component, row, column)
    component.Layout.Row = row; component.Layout.Column = column;
end

function out = blendHeat(modelDisplay, heatmap)
% The image the model graded with the map on top; opacity follows the map so that cold areas keep the retina visible.
    base = im2double(modelDisplay);
    heat = min(max(imresize(double(heatmap), [size(base, 1) size(base, 2)], 'bilinear'), 0), 1);
    colours = turbo(256);
    rgb = reshape(colours(round(heat * 255) + 1, :), [size(heat) 3]);
    alpha = 0.10 + 0.55 * heat;
    out = base .* (1 - alpha) + rgb .* alpha;
end

function text = shortStatus(status)
    if startsWith(status, "UNCALIBRATED")
        text = "Frozen (not calibrated here)";
    elseif startsWith(status, "site-calibrated")
        text = "Calibrated for this site";
    else
        text = status;
    end
end

function tf = haveLearned(r)
    tf = ~isempty(r) && isfield(r, 'learnedLesions') && isfield(r.learnedLesions, 'overlayPath') && ...
        strlength(string(r.learnedLesions.overlayPath)) > 0 && isfile(r.learnedLesions.overlayPath);
end

function text = learnedCounts(learned)
% "yellow exudates 12, red haemorrhages 3, ..." from the module's own summary.
    labels = struct('EX', "exudates (yellow)", 'HE', "haemorrhages (red)", 'MA', "microaneurysms (blue)", 'SE', "soft exudates (white)");   % overlay colours of runLearnedLesions
    names = string(learned.names); parts = strings(1, numel(names));
    for k = 1:numel(names)
        label = names(k); if isfield(labels, names(k)), label = labels.(names(k)); end
        if isstruct(learned.count), n = learned.count.(names(k)); else, n = learned.count(k); end
        parts(k) = sprintf('%s %d', label, n);
    end
    text = strjoin(parts, ', ');
end

function model = findLesionModel(baseDir, projectDir)
% The lesion segmenter is not distributed with the repository; use it when its two files are present locally.
    candidates = {fullfile(baseDir, 'models', 'lesion_unet_b2_1024.onnx'), fullfile(baseDir, 'models', 'lesion_thresholds.json'); ...
        fullfile(projectDir, 'results', 'model_improvement', 'lesion_onnx', 'lesion_unet_b2_1024.onnx'), ...
        fullfile(projectDir, 'results', 'model_improvement', 'x8_lock_candidate', 'thresholds.json')};
    model = [];
    for k = 1:size(candidates, 1)
        if isfile(candidates{k, 1}) && isfile(candidates{k, 2})
            model = struct('onnx', string(candidates{k, 1}), 'thresholds', string(candidates{k, 2}));
            return;
        end
    end
end

function closeDialog(dialog)
    if ~isempty(dialog) && isvalid(dialog), delete(dialog); end
end

function value = onOff(condition)
    if condition, value = 'on'; else, value = 'off'; end
end

function out = ternary(condition, a, b)
    if condition, out = a; else, out = b; end
end
