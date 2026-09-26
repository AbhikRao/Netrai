function digest = fileSHA256(path)
%FILESHA256 Stream content identity without relying on file timestamps.
    stream = fopen(path,'rb');
    if stream < 0, error('NetrAI:MissingArtifact','Cannot read %s',path); end
    cleanup = onCleanup(@() fclose(stream));
    hash = java.security.MessageDigest.getInstance('SHA-256');
    while true
        bytes = fread(stream,1024*1024,'*uint8');
        if isempty(bytes), break; end
        hash.update(typecast(bytes,'int8'));
    end
    digest = lower(reshape(dec2hex(typecast(hash.digest(),'uint8'),2)',1,[]));
end
