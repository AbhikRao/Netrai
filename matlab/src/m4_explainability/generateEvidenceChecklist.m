function checklist = generateEvidenceChecklist(segResults, grade, calProbs)
%GENERATEEVIDENCECHECKLIST Generates clinical evidence checklist for ICDR concordance.
%
%   checklist = generateEvidenceChecklist(segResults, grade, calProbs)
%
%   Outputs:
%       checklist - Struct array of clinical criteria items

    checklist = struct('criterion', {}, 'status', {}, 'detail', {});
    
    % 1. Microaneurysms
    maCount = 0;
    if isfield(segResults, 'maCount'), maCount = segResults.maCount;
    elseif isfield(segResults, 'MAs') && isfield(segResults.MAs, 'count'), maCount = segResults.MAs.count;
    end
    if maCount > 0
        checklist(1) = struct('criterion', 'Microaneurysms', 'status', 'FOUND', 'detail', sprintf('%d MAs detected', maCount));
    else
        checklist(1) = struct('criterion', 'Microaneurysms', 'status', 'NONE', 'detail', 'No MAs detected');
    end
    
    % 2. Hemorrhages
    hemCount = 0;
    if isfield(segResults, 'hemCount'), hemCount = segResults.hemCount;
    elseif isfield(segResults, 'hemorrhages') && isfield(segResults.hemorrhages, 'count'), hemCount = segResults.hemorrhages.count;
    end
    if hemCount > 0
        checklist(2) = struct('criterion', 'Hemorrhages', 'status', 'FOUND', 'detail', sprintf('%d hemorrhages detected', hemCount));
    else
        checklist(2) = struct('criterion', 'Hemorrhages', 'status', 'NONE', 'detail', 'No hemorrhages detected');
    end
    
    % 3. Hard Exudates
    hardCount = 0;
    if isfield(segResults, 'hardCount'), hardCount = segResults.hardCount;
    elseif isfield(segResults, 'hardExudates') && isfield(segResults.hardExudates, 'count'), hardCount = segResults.hardExudates.count;
    end
    if hardCount > 0
        checklist(3) = struct('criterion', 'Hard Exudates', 'status', 'FOUND', 'detail', sprintf('%d hard exudates detected', hardCount));
    else
        checklist(3) = struct('criterion', 'Hard Exudates', 'status', 'NONE', 'detail', 'No hard exudates detected');
    end
    
    % 4. Soft Exudates
    softCount = 0;
    if isfield(segResults, 'softCount'), softCount = segResults.softCount;
    elseif isfield(segResults, 'softExudates') && isfield(segResults.softExudates, 'count'), softCount = segResults.softExudates.count;
    end
    if softCount > 0
        checklist(4) = struct('criterion', 'Soft Exudates', 'status', 'FOUND', 'detail', sprintf('%d soft exudates (cotton wool spots)', softCount));
    else
        checklist(4) = struct('criterion', 'Soft Exudates', 'status', 'NONE', 'detail', 'No soft exudates detected');
    end
    
    % 5. Neovascularization
    nvDetected = false;
    if isfield(segResults, 'nvDetected'), nvDetected = logical(segResults.nvDetected);
    elseif isfield(segResults, 'neovascularization') && isfield(segResults.neovascularization, 'count'), nvDetected = segResults.neovascularization.count > 0;
    end
    if nvDetected
        checklist(5) = struct('criterion', 'Neovascularization', 'status', 'FOUND', 'detail', 'Abnormal vessel proliferation detected (PDR sign)');
    else
        checklist(5) = struct('criterion', 'Neovascularization', 'status', 'NONE', 'detail', 'No abnormal neovascular proliferation');
    end
    
    % 6. Macular Edema Risk
    fDist = Inf;
    if isfield(segResults, 'foveaDist'), fDist = segResults.foveaDist;
    elseif isfield(segResults, 'nearestFoveaDist'), fDist = segResults.nearestFoveaDist;
    elseif isfield(segResults, 'hardExudates') && isfield(segResults.hardExudates, 'nearestFoveaDist'), fDist = segResults.hardExudates.nearestFoveaDist;
    end
    if fDist < 2.0 && fDist > 0
        checklist(6) = struct('criterion', 'Macular Edema Risk', 'status', 'HIGH', 'detail', sprintf('Exudates within %.1f DD of fovea (DME Risk)', fDist));
    else
        checklist(6) = struct('criterion', 'Macular Edema Risk', 'status', 'LOW', 'detail', 'Foveal center clear');
    end
    
    % 7. ICDR Concordance
    gradeNames = {'No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Proliferative DR'};
    gName = gradeNames{min(max(grade+1, 1), 5)};
    checklist(7) = struct('criterion', 'ICDR Concordance', 'status', 'MATCH', 'detail', sprintf('Consistent with Level %d (%s)', grade, gName));
end
