classdef ReduceMeanLayer1029 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1029';
        end
    end


    methods
        function this = ReduceMeanLayer1029(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone_432'};
        end

        function [x_model_backbone_432] = predict(this, x_model_backbone_426)
            if isdlarray(x_model_backbone_426)
                x_model_backbone_426 = stripdims(x_model_backbone_426);
            end
            x_model_backbone_426NumDims = 4;
            x_model_backbone_426 = netrai_fold0.ops.permuteInputVar(x_model_backbone_426, [4 3 1 2], 4);

            [x_model_backbone_432, x_model_backbone_432NumDims] = ReduceMeanGraph1087(this, x_model_backbone_426, x_model_backbone_426NumDims, false);
            x_model_backbone_432 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_432, [3 4 2 1], 4);

            x_model_backbone_432 = dlarray(single(x_model_backbone_432), 'SSCB');
        end

        function [x_model_backbone_432] = forward(this, x_model_backbone_426)
            if isdlarray(x_model_backbone_426)
                x_model_backbone_426 = stripdims(x_model_backbone_426);
            end
            x_model_backbone_426NumDims = 4;
            x_model_backbone_426 = netrai_fold0.ops.permuteInputVar(x_model_backbone_426, [4 3 1 2], 4);

            [x_model_backbone_432, x_model_backbone_432NumDims] = ReduceMeanGraph1087(this, x_model_backbone_426, x_model_backbone_426NumDims, true);
            x_model_backbone_432 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_432, [3 4 2 1], 4);

            x_model_backbone_432 = dlarray(single(x_model_backbone_432), 'SSCB');
        end

        function [x_model_backbone_432, x_model_backbone_432NumDims1089] = ReduceMeanGraph1087(this, x_model_backbone_426, x_model_backbone_426NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1088, x_model_backbone_426NumDims);
            xMean = mean(x_model_backbone_426, dims);
            x_model_backbone_432 = xMean;
            x_model_backbone_432NumDims = x_model_backbone_426NumDims;

            % Set graph output arguments
            x_model_backbone_432NumDims1089 = x_model_backbone_432NumDims;

        end

    end

end