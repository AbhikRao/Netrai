function manifest = writeDeploymentBundle(modelPath,configPath)
%WRITEDEPLOYMENTBUNDLE Explicitly bind a reviewed model/config/source release.
% Run after an intentional deployment change, not to bypass an integrity error.
% This records artifact identity; it does not establish clinical acceptance.
    baseDir=fileparts(mfilename('fullpath'));
    if nargin<1, modelPath=fullfile(baseDir,'models','netrai_fold0.onnx'); end
    if nargin<2, configPath=fullfile(baseDir,'models','netrai_config.json'); end
    addpath(genpath(fullfile(baseDir,'src')));
    base=fileparts(configPath);
    [~,m,e]=fileparts(modelPath); [~,c,f]=fileparts(configPath);
    roles=["model","config","preprocessing","grading","calibration","routing","decoding"];
    paths=[string(m)+e,string(c)+f,"../src/m3_grading/preprocessModelInput.m", ...
        "../src/m3_grading/predictDRGrade.m", ...
        "../src/m4_explainability/calibrateConfidence.m", ...
        "../src/utils/assignScreeningTriage.m","../src/utils/readScreeningImage.m"];
    artifacts=repmat(struct('role','','path','','sha256',''),numel(roles),1);
    for i=1:numel(roles)
        artifacts(i)=struct('role',char(roles(i)),'path',char(paths(i)), ...
            'sha256',fileSHA256(fullfile(base,paths(i))));
    end
    manifest=struct('schema_version',2,'artifacts',artifacts);
    path=fullfile(base,'netrai_bundle.json'); temporary=string(tempname(base))+'.json';
    fid=fopen(temporary,'w');
    if fid<0, error('NetrAI:WriteFailed','Cannot create deployment bundle.'); end
    cleanup=onCleanup(@() fclose(fid));
    fprintf(fid,'%s\n',jsonencode(manifest,'PrettyPrint',true)); clear cleanup;
    verifyDeploymentBundle(string(modelPath),string(configPath),temporary);
    [ok,message]=movefile(temporary,path,'f');
    if ~ok, error('NetrAI:WriteFailed','Cannot publish deployment bundle: %s',message); end
end
