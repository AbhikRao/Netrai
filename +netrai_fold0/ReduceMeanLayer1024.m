classdef ReduceMeanLayer1024 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
    end

    properties (State)
    end

    properties
        Vars
        NumDims
    end


    methods(Static, Hidden)
        % Specify the path to the class that will be used for codegen
        function name = matlabCodegenRedirect(~)
            name = 'netrai_fold0.coder.ReduceMeanLayer1024';
        end
    end


    methods
        function this = ReduceMeanLayer1024(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone_357'};
        end

        function [x_model_backbone_357] = predict(this, x_model_backbone_351)
            if isdlarray(x_model_backbone_351)
                x_model_backbone_351 = stripdims(x_model_backbone_351);
            end
            x_model_backbone_351NumDims = 4;
            x_model_backbone_351 = netrai_fold0.ops.permuteInputVar(x_model_backbone_351, [4 3 1 2], 4);

            [x_model_backbone_357, x_model_backbone_357NumDims] = ReduceMeanGraph1072(this, x_model_backbone_351, x_model_backbone_351NumDims, false);
            x_model_backbone_357 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_357, [3 4 2 1], 4);

            x_model_backbone_357 = dlarray(single(x_model_backbone_357), 'SSCB');
        end

        function [x_model_backbone_357] = forward(this, x_model_backbone_351)
            if isdlarray(x_model_backbone_351)
                x_model_backbone_351 = stripdims(x_model_backbone_351);
            end
            x_model_backbone_351NumDims = 4;
            x_model_backbone_351 = netrai_fold0.ops.permuteInputVar(x_model_backbone_351, [4 3 1 2], 4);

            [x_model_backbone_357, x_model_backbone_357NumDims] = ReduceMeanGraph1072(this, x_model_backbone_351, x_model_backbone_351NumDims, true);
            x_model_backbone_357 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_357, [3 4 2 1], 4);

            x_model_backbone_357 = dlarray(single(x_model_backbone_357), 'SSCB');
        end

        function [x_model_backbone_357, x_model_backbone_357NumDims1074] = ReduceMeanGraph1072(this, x_model_backbone_351, x_model_backbone_351NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1073, x_model_backbone_351NumDims);
            xMean = mean(x_model_backbone_351, dims);
            x_model_backbone_357 = xMean;
            x_model_backbone_357NumDims = x_model_backbone_351NumDims;

            % Set graph output arguments
            x_model_backbone_357NumDims1074 = x_model_backbone_357NumDims;

        end

    end

end