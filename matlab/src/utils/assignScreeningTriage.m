function action = assignScreeningTriage(qualityRejected, gradeReferable, referableHeadPositive)
%ASSIGNSCREENINGTRIAGE Fail-safe four-way screening disposition.

    if qualityRejected
        action = "recapture";
    elseif logical(gradeReferable) ~= logical(referableHeadPositive)
        action = "uncertain_human_review";
    elseif gradeReferable
        action = "refer";
    else
        action = "auto_clear";
    end
end
