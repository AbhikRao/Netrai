function metrics = assessGradCAMQuality(heatmap, fovMask)
%ASSESSGRADCAMQUALITY Heuristic border/corner/FOV shortcut indicators.
% These are review flags, not clinical validation or proof of model error.

    if ~(isnumeric(heatmap) || islogical(heatmap)) || ~isreal(heatmap) ...
            || ~ismatrix(heatmap) || isempty(heatmap) ...
            || any(~isfinite(heatmap(:))) || any(heatmap(:) < 0)
        error('NetrAI:InvalidGradCAM', 'Grad-CAM must be a finite nonnegative real 2-D array.');
    end
    if nargin < 2 || ~(isnumeric(fovMask) || islogical(fovMask)) ...
            || ~isreal(fovMask) || ~ismatrix(fovMask) || isempty(fovMask) ...
            || any(~isfinite(fovMask(:))) || any(fovMask(:) ~= 0 & fovMask(:) ~= 1)
        error('NetrAI:InvalidFOV', 'FOV must be a finite real binary 2-D mask.');
    end
    cam = double(heatmap);
    [h,w] = size(cam); total = sum(cam,'all'); band = max(1,round(min(h,w)*0.10));
    if ~isfinite(total)
        error('NetrAI:InvalidGradCAM', 'Grad-CAM total attention must be finite.');
    end
    fov=imresize(logical(fovMask),[h w],'nearest');
    border = false(h,w); border(1:band,:)=true; border(end-band+1:end,:)=true;
    border(:,1:band)=true; border(:,end-band+1:end)=true;
    if total <= 0
        metrics = struct('border_attention_fraction',0,'border_enrichment',0, ...
            'max_corner_enrichment',0,'top5_border_fraction',0, ...
            'outside_fov_attention_fraction',0,'top20_outside_fov_fraction',0, ...
            'shortcut_flag',true,'informative',false,'status',"zero_attribution");
        return;
    end
    informative=any(~border(:));
    borderEnrichment=0;
    if informative
        borderMean=mean(cam(border)); interiorMean=mean(cam(~border));
        borderEnrichment=borderMean/max(interiorMean,eps);
    end
    side=max(1,round(min(h,w)*0.15)); globalMean=mean(cam,'all');
    cornerMeans=[mean(cam(1:side,1:side),'all'),mean(cam(1:side,end-side+1:end),'all'), ...
        mean(cam(end-side+1:end,1:side),'all'),mean(cam(end-side+1:end,end-side+1:end),'all')];
    cornerEnrichment=max(cornerMeans)/max(globalMean,eps);
    values=sort(cam(:)); top5Threshold=values(max(1,ceil(0.95*numel(values))));
    top5=cam>=top5Threshold; top5Border=mean(border(top5));
    outsideFraction=sum(cam(~fov))/total;
    top20Threshold=values(max(1,ceil(0.80*numel(values)))); top20=cam>=top20Threshold;
    top20Outside=mean(~fov(top20));
    flag=~informative || borderEnrichment>1.5 || cornerEnrichment>2.0 || top5Border>0.60 ...
        || outsideFraction>0.20 || top20Outside>0.25;
    status="available";
    if ~informative, status="insufficient_spatial_grid"; end
    metrics=struct('border_attention_fraction',sum(cam(border))/total, ...
        'border_enrichment',borderEnrichment,'max_corner_enrichment',cornerEnrichment, ...
        'top5_border_fraction',top5Border,'outside_fov_attention_fraction',outsideFraction, ...
        'top20_outside_fov_fraction',top20Outside,'shortcut_flag',flag, ...
        'informative',informative,'status',status);
end
