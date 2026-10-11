function [image, boundPath, cleanup] = readBoundValidationImage(path, expectedDigest)
%READBOUNDVALIDATIONIMAGE Decode exactly the bytes named by the cohort snapshot.
    stream = fopen(path,'rb');
    if stream<0, error('NetrAI:MissingArtifact','Cannot read image %s.',path); end
    closeStream = onCleanup(@() fclose(stream)); %#ok<NASGU>
    bytes = fread(stream,inf,'*uint8');
    if ~strcmp(sha256Bytes(bytes),expectedDigest)
        error('NetrAI:ChangedInput','Image bytes differ from the cohort snapshot: %s.',path);
    end
    [~,~,ext]=fileparts(path); boundPath=string(tempname)+ext;
    output=fopen(boundPath,'wb');
    if output<0, error('NetrAI:WriteFailed','Cannot stage image snapshot.'); end
    closeOutput=onCleanup(@() fclose(output));
    cleanup=onCleanup(@() delete(boundPath));
    count=fwrite(output,bytes,'uint8'); clear closeOutput;
    if count~=numel(bytes), error('NetrAI:WriteFailed','Incomplete image snapshot.'); end
    image=readScreeningImage(boundPath);
end
