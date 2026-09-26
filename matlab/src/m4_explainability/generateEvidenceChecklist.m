function checklist = generateEvidenceChecklist(segResults, grade, calProbs)
%GENERATEEVIDENCECHECKLIST Unverified research candidates, never diagnoses.
    checklist = struct('criterion', {}, 'status', {}, 'detail', {});
    names = {'MA candidates','Hemorrhage candidates','Hard exudate candidates','Soft exudate candidates'};
    fields = {'maCount','hemCount','hardCount','softCount'};
    for k = 1:4
        count = NaN;
        if isfield(segResults, fields{k}), count = segResults.(fields{k}); end
        if isfinite(count)
            status = 'CANDIDATE';
            detail = sprintf('%d unverified; zero does not exclude lesions', count);
        else
            status = 'UNKNOWN'; detail = 'Not assessed';
        end
        checklist(k) = struct('criterion', names{k}, 'status', status, 'detail', detail);
    end
    distance = NaN;
    if isfield(segResults, 'foveaDist'), distance = segResults.foveaDist; end
    status = 'UNKNOWN'; detail = 'Not assessable; no DME conclusion';
    if isfinite(distance) && distance >= 0
        status = 'CANDIDATE';
        detail = sprintf('%.2f disc diameters; not a DME diagnosis', distance);
    end
    checklist(5) = struct('criterion','Exudate-fovea geometry','status',status,'detail',detail);
    detail = 'Not assessed'; status = 'UNKNOWN';
    if isfield(segResults, 'nvDetected')
        status = 'CANDIDATE';
        if segResults.nvDetected
            detail = 'Heuristic flagged; needs expert confirmation';
        else
            detail = 'Not flagged; NV not excluded';
        end
    end
    checklist(6) = struct('criterion','Vessel-proliferation heuristic','status',status,'detail',detail);
    checklist(7) = struct('criterion','Grade attribution','status','UNVERIFIED', ...
        'detail',sprintf('Grade %d; lesion concordance not established',grade));
end
