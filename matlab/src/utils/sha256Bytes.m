function digest = sha256Bytes(bytes)
%SHA256BYTES Hash an in-memory byte snapshot, without timestamp assumptions.
    hash = java.security.MessageDigest.getInstance('SHA-256');
    hash.update(typecast(uint8(bytes(:)),'int8'));
    digest = lower(reshape(dec2hex(typecast(hash.digest(),'uint8'),2)',1,[]));
end
