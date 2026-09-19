function lgraph = buildVesselUNet(inputSize)
%BUILDVESSELUNET Builds a U-Net architecture for vessel segmentation.
%   lgraph = buildVesselUNet(inputSize)
%   Inputs:
%       inputSize - Size of input image [H W C]
%   Outputs:
%       lgraph - LayerGraph containing the U-Net architecture

    if nargin < 1
        inputSize = [512 512 1];
    end
    
    % Check if unetLayers is available
    if exist('unetLayers', 'file')
        try
            lgraph = unetLayers(inputSize, 2, 'EncoderDepth', 3);
            return;
        catch
            % Fall back to manual creation
        end
    end
    
    % Manual U-Net building (simplified)
    lgraph = layerGraph();
    
    % Input
    layers = [
        imageInputLayer(inputSize, 'Name', 'input')
        
        % Encoder Level 1
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'enc1_conv')
        batchNormalizationLayer('Name', 'enc1_bn')
        reluLayer('Name', 'enc1_relu')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'enc1_pool')
        
        % Encoder Level 2
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'enc2_conv')
        batchNormalizationLayer('Name', 'enc2_bn')
        reluLayer('Name', 'enc2_relu')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'enc2_pool')
        
        % Encoder Level 3
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'enc3_conv')
        batchNormalizationLayer('Name', 'enc3_bn')
        reluLayer('Name', 'enc3_relu')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'enc3_pool')
        
        % Bridge
        convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'bridge_conv')
        batchNormalizationLayer('Name', 'bridge_bn')
        reluLayer('Name', 'bridge_relu')
    ];
    lgraph = addLayers(lgraph, layers);
    
    % Decoder Level 3
    dec3Layers = [
        transposedConv2dLayer(2, 128, 'Stride', 2, 'Name', 'dec3_up')
    ];
    lgraph = addLayers(lgraph, dec3Layers);
    lgraph = connectLayers(lgraph, 'bridge_relu', 'dec3_up');
    
    dec3ConcatConv = [
        depthConcatenationLayer(2, 'Name', 'dec3_concat')
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'dec3_conv')
    ];
    lgraph = addLayers(lgraph, dec3ConcatConv);
    lgraph = connectLayers(lgraph, 'dec3_up', 'dec3_concat/in1');
    lgraph = connectLayers(lgraph, 'enc3_relu', 'dec3_concat/in2');
    
    % Decoder Level 2
    dec2Layers = [
        transposedConv2dLayer(2, 64, 'Stride', 2, 'Name', 'dec2_up')
    ];
    lgraph = addLayers(lgraph, dec2Layers);
    lgraph = connectLayers(lgraph, 'dec3_conv', 'dec2_up');
    
    dec2ConcatConv = [
        depthConcatenationLayer(2, 'Name', 'dec2_concat')
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'dec2_conv')
    ];
    lgraph = addLayers(lgraph, dec2ConcatConv);
    lgraph = connectLayers(lgraph, 'dec2_up', 'dec2_concat/in1');
    lgraph = connectLayers(lgraph, 'enc2_relu', 'dec2_concat/in2');
    
    % Decoder Level 1
    dec1Layers = [
        transposedConv2dLayer(2, 32, 'Stride', 2, 'Name', 'dec1_up')
    ];
    lgraph = addLayers(lgraph, dec1Layers);
    lgraph = connectLayers(lgraph, 'dec2_conv', 'dec1_up');
    
    dec1ConcatConv = [
        depthConcatenationLayer(2, 'Name', 'dec1_concat')
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'dec1_conv')
        
        % Output
        convolution2dLayer(1, 2, 'Name', 'out_conv')
        softmaxLayer('Name', 'softmax')
        pixelClassificationLayer('Name', 'pixelClass')
    ];
    lgraph = addLayers(lgraph, dec1ConcatConv);
    lgraph = connectLayers(lgraph, 'dec1_up', 'dec1_concat/in1');
    lgraph = connectLayers(lgraph, 'enc1_relu', 'dec1_concat/in2');
end
