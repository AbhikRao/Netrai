classdef ReduceMeanLayer1019 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1019';
        end
    end


    methods
        function this = ReduceMeanLayer1019(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone_283'};
        end

        function [x_model_backbone_283] = predict(this, x_model_backbone_277)
            if isdlarray(x_model_backbone_277)
                x_model_backbone_277 = stripdims(x_model_backbone_277);
            end
            x_model_backbone_277NumDims = 4;
            x_model_backbone_277 = netrai_fold0.ops.permuteInputVar(x_model_backbone_277, [4 3 1 2], 4);

            [x_model_backbone_283, x_model_backbone_283NumDims] = ReduceMeanGraph1057(this, x_model_backbone_277, x_model_backbone_277NumDims, false);
            x_model_backbone_283 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_283, [3 4 2 1], 4);

            x_model_backbone_283 = dlarray(single(x_model_backbone_283), 'SSCB');
        end

        function [x_model_backbone_283] = forward(this, x_model_backbone_277)
            if isdlarray(x_model_backbone_277)
                x_model_backbone_277 = stripdims(x_model_backbone_277);
            end
            x_model_backbone_277NumDims = 4;
            x_model_backbone_277 = netrai_fold0.ops.permuteInputVar(x_model_backbone_277, [4 3 1 2], 4);

            [x_model_backbone_283, x_model_backbone_283NumDims] = ReduceMeanGraph1057(this, x_model_backbone_277, x_model_backbone_277NumDims, true);
            x_model_backbone_283 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_283, [3 4 2 1], 4);

            x_model_backbone_283 = dlarray(single(x_model_backbone_283), 'SSCB');
        end

        function [x_model_backbone_283, x_model_backbone_283NumDims1059] = ReduceMeanGraph1057(this, x_model_backbone_277, x_model_backbone_277NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1058, x_model_backbone_277NumDims);
            xMean = mean(x_model_backbone_277, dims);
            x_model_backbone_283 = xMean;
            x_model_backbone_283NumDims = x_model_backbone_277NumDims;

            % Set graph output arguments
            x_model_backbone_283NumDims1059 = x_model_backbone_283NumDims;

        end

    end

end