function [modelInput, modelFov] = preprocessModelInput(imageRGB, config)
%PREPROCESSMODELINPUT Reproduce the released legacy-area-v1 input contract.
% Input is uint8 RGB (or grayscale); output is single normalized RGB.
% Pixel-area resizing, uint8 fixed-point Gaussian with REFLECT_101 borders,
% and the integer blend preserve the released OpenCV operator semantics.

    required={'input_size','black_crop_threshold','ben_graham_sigma', ...
        'imagenet_mean','imagenet_std'};
    if ~isstruct(config) || ~isscalar(config) || ~all(isfield(config,required))
        error('NetrAI:InvalidConfig','Missing model input configuration.');
    end
    if isfield(config,'preprocessing_version') && ...
            string(config.preprocessing_version)~="legacy-area-v1"
        error('NetrAI:InvalidConfig','Unsupported model preprocessing version.');
    end
    if ~isnumeric(config.input_size) || ~isscalar(config.input_size) || ...
            ~isfinite(config.input_size) || config.input_size<1 || mod(config.input_size,1)~=0 || ...
            ~isnumeric(config.black_crop_threshold) || ~isscalar(config.black_crop_threshold) || ...
            ~isfinite(config.black_crop_threshold) || config.black_crop_threshold<0 || ...
            config.black_crop_threshold>255 || ...
            ~isnumeric(config.ben_graham_sigma) || ~isscalar(config.ben_graham_sigma) || ...
            config.ben_graham_sigma~=10 || ...
            ~isnumeric(config.imagenet_mean) || numel(config.imagenet_mean)~=3 || ...
            any(~isfinite(config.imagenet_mean(:))) || ...
            ~isnumeric(config.imagenet_std) || numel(config.imagenet_std)~=3 || ...
            any(~isfinite(config.imagenet_std(:)) | config.imagenet_std(:)<=0)
        error('NetrAI:InvalidConfig','Invalid legacy-area-v1 input configuration.');
    end
    if isempty(imageRGB) || ~isa(imageRGB, 'uint8')
        error('NetrAI:InvalidImage','Expected a nonempty uint8 RGB image.');
    end
    if ismatrix(imageRGB), imageRGB=repmat(imageRGB,1,1,3); end
    if ndims(imageRGB)~=3 || size(imageRGB,3)~=3
        error('NetrAI:InvalidImage','Expected H-by-W-by-3 RGB.');
    end
    foreground=max(imageRGB,[],3)>double(config.black_crop_threshold);
    if any(foreground(:))
        rows=find(any(foreground,2)); cols=find(any(foreground,1));
        imageRGB=imageRGB(rows(1):rows(end),cols(1):cols(end),:);
        foreground=foreground(rows(1):rows(end),cols(1):cols(end));
    end
    target=double(config.input_size);
    resized=areaResize(imageRGB,target,target);
    blurred=fixedGaussian(resized,double(config.ben_graham_sigma));
    enhanced=uint8(min(255,max(0,4*double(resized)-4*double(blurred)+128)));
    means=reshape(single(config.imagenet_mean),1,1,3);
    stds=reshape(single(config.imagenet_std),1,1,3);
    modelInput=(single(enhanced)./single(255)-means)./stds;
    ys=min(size(foreground,1),floor((0:target-1)*size(foreground,1)/target)+1);
    xs=min(size(foreground,2),floor((0:target-1)*size(foreground,2)/target)+1);
    modelFov=foreground(ys,xs);
end

function resized=areaResize(image,height,width)
    sourceHeight=size(image,1); sourceWidth=size(image,2);
    scaleY=sourceHeight/height; scaleX=sourceWidth/width;
    if sourceHeight==height && sourceWidth==width
        resized=image; return;
    end
    if scaleX>=1 && scaleY>=1
        if scaleX==floor(scaleX) && scaleY==floor(scaleY)
            values=zeros(height,width,3);
            for y=0:scaleY-1
                for x=0:scaleX-1
                    values=values+double(image(y+1:scaleY:end,x+1:scaleX:end,:));
                end
            end
            if scaleX==2 && scaleY==2
                resized=uint8(floor((values+2)/4));
            else
                resized=uint8(roundEven(single(values).*single(1/(scaleX*scaleY))));
            end
            return;
        end
        [xIndices,xWeights]=areaWeights(sourceWidth,width);
        horizontal=zeros(sourceHeight,width,3,'single');
        for k=1:size(xIndices,2)
            horizontal=horizontal+single(image(:,xIndices(:,k),:)).*reshape(xWeights(:,k),1,width,1);
        end
        [yIndices,yWeights]=areaWeights(sourceHeight,height);
        values=zeros(height,width,3,'single');
        for k=1:size(yIndices,2)
            values=values+horizontal(yIndices(:,k),:,:).*reshape(yWeights(:,k),height,1,1);
        end
        resized=uint8(roundEven(values));
    else
        [x,ax]=upWeights(sourceWidth,width);
        [y,ay]=upWeights(sourceHeight,height);
        horizontal=double(image(:,x,:)).*reshape(2048-ax,1,width,1) ...
            +double(image(:,min(x+1,sourceWidth),:)).*reshape(ax,1,width,1);
        horizontal=floor(horizontal/16);
        upper=floor(horizontal(y,:,:).*reshape(2048-ay,height,1,1)/65536);
        lower=floor(horizontal(min(y+1,sourceHeight),:,:).*reshape(ay,height,1,1)/65536);
        resized=uint8(floor((upper+lower+2)/4));
    end
end

function [indices,weights]=areaWeights(source,target)
    scale=source/target;
    indices=ones(target,ceil(scale)+2);
    weights=zeros(size(indices),'single');
    for dest=0:target-1
        left=dest*scale; right=left+scale;
        denominator=min(scale,source-left);
        first=min(ceil(left),source-1); last=min(floor(right),source-1);
        first=min(first,last); k=1;
        if first-left>1e-3
            indices(dest+1,k)=first;
            weights(dest+1,k)=single((first-left)/denominator); k=k+1;
        end
        for src=first:last-1
            indices(dest+1,k)=src+1;
            weights(dest+1,k)=single(1/denominator); k=k+1;
        end
        if right-last>1e-3
            indices(dest+1,k)=last+1;
            weights(dest+1,k)=single(min([right-last,1,denominator])/denominator);
        end
    end
end

function [index,weight]=upWeights(source,target)
    ratio=source/target; inverse=target/source;
    coordinate=floor((0:target-1)*ratio);
    fraction=single((1:target)-(coordinate+1)*inverse);
    positive=fraction>0;
    fraction(~positive)=0;
    fraction(positive)=fraction(positive)-floor(fraction(positive));
    fraction(coordinate>=source-1)=0;
    index=min(source,coordinate+1);
    weight=double(roundEven(fraction*single(2048)));
end

function blurred=fixedGaussian(image,sigma)
    radius=(bitor(round(6*sigma+1),1)-1)/2;
    positions=-radius:radius;
    kernel=exp(-(positions.^2)/(2*sigma^2)); kernel=kernel/sum(kernel);
    coefficients=zeros(size(kernel)); residual=0;
    for k=1:radius
        adjusted=256*kernel(k)+residual;
        coefficients(k)=roundEven(adjusted);
        coefficients(end+1-k)=coefficients(k);
        residual=adjusted-coefficients(k);
    end
    coefficients(radius+1)=256-sum(coefficients);
    ys=reflectIndices(size(image,1),radius);
    xs=reflectIndices(size(image,2),radius);
    blurred=zeros(size(image),'uint8');
    for channel=1:3
        horizontal=conv2(double(image(:,xs,channel)),coefficients,'valid');
        vertical=conv2(horizontal(ys,:),coefficients','valid');
        blurred(:,:,channel)=uint8(floor((vertical+32768)/65536));
    end
end

function index=reflectIndices(length,radius)
    if length==1
        index=ones(1,length+2*radius); return;
    end
    period=2*(length-1);
    coordinate=mod(-radius:length+radius-1,period);
    index=min(coordinate,period-coordinate)+1;
end

function rounded=roundEven(value)
    lower=floor(value); fraction=value-lower;
    rounded=lower+(fraction>0.5 | (fraction==0.5 & mod(lower,2)==1));
end
