function members = referralHeadMembers(net)
%REFERRALHEADMEMBERS The members of a NetrAI network (one for the legacy model, two for E_768): for each, the name of its last
% activation (the input of its global average pooling) and its head layers. Used by referralGradCAM and ForwardRecorder.
    layers = net.Layers;                % one copy: every net.Layers access rebuilds the layer array (about 0.1 s for E_768)
    names = string({layers.Name});
    pools = find(arrayfun(@(l) isa(l, 'nnet.cnn.layer.GlobalAveragePooling2DLayer'), layers))';
    if isempty(pools)
        error('NetrAI:GradCAMHeadMismatch', 'The network has no global average pooling layer; it is not a NetrAI model.');
    end
    src = string(net.Connections.Source); dst = extractBefore(string(net.Connections.Destination) + "/", "/");
    members = struct('activation', {}, 'head', {});
    for g = pools
        feature = src(dst == names(g));
        members(end + 1) = struct('activation', feature(1), 'head', headLayers(layers, names, src, dst, names(g))); %#ok<AGROW>
    end
end

function head = headLayers(allLayers, names, src, dst, pool)
% Fully connected and batch-norm layers after one member's pooling layer, in graph order, identified by their sizes.
    order = strings(0, 1); frontier = pool; seen = pool;
    while ~isempty(frontier)
        nextLayers = unique(dst(ismember(src, frontier)), 'stable');
        nextLayers = nextLayers(~ismember(nextLayers, seen));
        order = [order; nextLayers(:)]; %#ok<AGROW>
        seen = [seen; nextLayers(:)]; %#ok<AGROW>
        frontier = nextLayers;
    end
    layers = arrayfun(@(n) allLayers(names == n), order, 'UniformOutput', false);   % mixed layer classes
    fc = [layers{cellfun(@(l) isa(l, 'nnet.cnn.layer.FullyConnectedLayer'), layers)}];
    bn = [layers{cellfun(@(l) isa(l, 'nnet.cnn.layer.BatchNormalizationLayer'), layers)}];
    pick = @(set, out) set(arrayfun(@(l) size(l.Weights, 1) == out, set));
    fc1 = pick(fc, 512); fc2 = pick(fc, 256); ref = pick(fc, 1);
    if numel(fc1) ~= 1 || numel(fc2) ~= 1 || numel(ref) ~= 1 || numel(bn) ~= 2
        error('NetrAI:GradCAMHeadMismatch', 'Unexpected member head structure.');
    end
    head = struct('fc1', fc1, 'bn1', bn(arrayfun(@(l) numel(l.TrainedMean) == 512, bn)), 'fc2', fc2, ...
        'bn2', bn(arrayfun(@(l) numel(l.TrainedMean) == 256, bn)), 'ref', ref);
end
