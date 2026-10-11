function resolved = verifyRuntimeDeploymentSources(manifest)
%VERIFYRUNTIMEDEPLOYMENTSOURCES Bind callable MATLAB functions to source roles.
% File-only bundle checks may inspect a relocated deployment. Inference must
% also use the exact bytes selected by MATLAB's current function resolution.
    arguments
        manifest (1,1) struct
    end
    roles=["preprocessing","grading","calibration","routing","decoding"];
    functions=["preprocessModelInput","predictDRGrade","calibrateConfidence", ...
        "assignScreeningTriage","readScreeningImage"];
    if ~isfield(manifest,'artifacts') || ~isstruct(manifest.artifacts) || ...
            ~all(isfield(manifest.artifacts,{'role','sha256'}))
        error('NetrAI:InvalidBundle','Runtime source verification requires bound source roles.');
    end
    actualRoles=string({manifest.artifacts.role});
    resolved=repmat(struct('role','','function_name','','path','','sha256',''),numel(roles),1);
    for index=1:numel(roles)
        entries=manifest.artifacts(actualRoles==roles(index));
        if numel(entries)~=1 || ~ischar(entries.sha256) || ...
                isempty(regexp(entries.sha256,'^[0-9a-f]{64}$','once'))
            error('NetrAI:InvalidBundle','Runtime source role %s must be bound exactly once.',roles(index));
        end
        resolvedPath=string(which(functions(index)));
        if strlength(resolvedPath)==0 || ~isfile(resolvedPath)
            error('NetrAI:RuntimeSourceMissing', ...
                'MATLAB cannot resolve the bound deployment function %s.',functions(index));
        end
        digest=fileSHA256(resolvedPath);
        if ~strcmp(digest,entries.sha256)
            error('NetrAI:RuntimeSourceMismatch', ...
                'MATLAB resolves %s to source bytes outside the bound %s role: %s.', ...
                functions(index),roles(index),resolvedPath);
        end
        resolved(index)=struct('role',char(roles(index)), ...
            'function_name',char(functions(index)),'path',char(resolvedPath),'sha256',digest);
    end
end
