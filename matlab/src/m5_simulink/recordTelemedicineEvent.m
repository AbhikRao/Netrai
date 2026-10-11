function records = recordTelemedicineEvent(eventCode, patientId, attempt, route)
%RECORDTELEMEDICINEEVENT Optional small-fixture identity ledger; no policy.
% Zero resets, negative reads. Positive appends [event PatientId Attempt Route].
    persistent ledger;
    if isempty(ledger), ledger = zeros(0, 4); end
    if eventCode == 0
        ledger = zeros(0, 4);
    elseif eventCode > 0
        ledger(end+1, :) = double([eventCode, patientId, attempt, route]);
    end
    records = ledger;
end
