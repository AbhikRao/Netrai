function digests = textSHA256(values)
%TEXTSHA256 Lower-case hex SHA-256 of the UTF-8 text of each value (same digests as text_sha256 in python/calibrate_site.py).
% Site policies record patient identities only in this hashed form.
    arguments
        values string
    end
    digests = strings(size(values));
    for i = 1:numel(values)
        hash = java.security.MessageDigest.getInstance('SHA-256');
        bytes = unicode2native(char(values(i)), 'UTF-8');
        if ~isempty(bytes)
            hash.update(typecast(uint8(bytes), 'int8'));
        end
        digests(i) = string(lower(reshape(dec2hex(typecast(hash.digest(), 'uint8'), 2)', 1, [])));
    end
end
