classdef ReduceMeanLayer1027 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1027';
        end
    end


    methods
        function this = ReduceMeanLayer1027(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone_402'};
        end

        function [x_model_backbone_402] = predict(this, x_model_backbone_396)
            if isdlarray(x_model_backbone_396)
                x_model_backbone_396 = stripdims(x_model_backbone_396);
            end
            x_model_backbone_396NumDims = 4;
            x_model_backbone_396 = netrai_fold0.ops.permuteInputVar(x_model_backbone_396, [4 3 1 2], 4);

            [x_model_backbone_402, x_model_backbone_402NumDims] = ReduceMeanGraph1081(this, x_model_backbone_396, x_model_backbone_396NumDims, false);
            x_model_backbone_402 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_402, [3 4 2 1], 4);

            x_model_backbone_402 = dlarray(single(x_model_backbone_402), 'SSCB');
        end

        function [x_model_backbone_402] = forward(this, x_model_backbone_396)
            if isdlarray(x_model_backbone_396)
                x_model_backbone_396 = stripdims(x_model_backbone_396);
            end
            x_model_backbone_396NumDims = 4;
            x_model_backbone_396 = netrai_fold0.ops.permuteInputVar(x_model_backbone_396, [4 3 1 2], 4);

            [x_model_backbone_402, x_model_backbone_402NumDims] = ReduceMeanGraph1081(this, x_model_backbone_396, x_model_backbone_396NumDims, true);
            x_model_backbone_402 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_402, [3 4 2 1], 4);

            x_model_backbone_402 = dlarray(single(x_model_backbone_402), 'SSCB');
        end

        function [x_model_backbone_402, x_model_backbone_402NumDims1083] = ReduceMeanGraph1081(this, x_model_backbone_396, x_model_backbone_396NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1082, x_model_backbone_396NumDims);
            xMean = mean(x_model_backbone_396, dims);
            x_model_backbone_402 = xMean;
            x_model_backbone_402NumDims = x_model_backbone_396NumDims;

            % Set graph output arguments
            x_model_backbone_402NumDims1083 = x_model_backbone_402NumDims;

        end

    end

end