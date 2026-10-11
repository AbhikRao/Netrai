classdef ReduceMeanLayer1005 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1005';
        end
    end


    methods
        function this = ReduceMeanLayer1005(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone__76'};
        end

        function [x_model_backbone__76] = predict(this, x_model_backbone__70)
            if isdlarray(x_model_backbone__70)
                x_model_backbone__70 = stripdims(x_model_backbone__70);
            end
            x_model_backbone__70NumDims = 4;
            x_model_backbone__70 = netrai_fold0.ops.permuteInputVar(x_model_backbone__70, [4 3 1 2], 4);

            [x_model_backbone__76, x_model_backbone__76NumDims] = ReduceMeanGraph1015(this, x_model_backbone__70, x_model_backbone__70NumDims, false);
            x_model_backbone__76 = netrai_fold0.ops.permuteOutputVar(x_model_backbone__76, [3 4 2 1], 4);

            x_model_backbone__76 = dlarray(single(x_model_backbone__76), 'SSCB');
        end

        function [x_model_backbone__76] = forward(this, x_model_backbone__70)
            if isdlarray(x_model_backbone__70)
                x_model_backbone__70 = stripdims(x_model_backbone__70);
            end
            x_model_backbone__70NumDims = 4;
            x_model_backbone__70 = netrai_fold0.ops.permuteInputVar(x_model_backbone__70, [4 3 1 2], 4);

            [x_model_backbone__76, x_model_backbone__76NumDims] = ReduceMeanGraph1015(this, x_model_backbone__70, x_model_backbone__70NumDims, true);
            x_model_backbone__76 = netrai_fold0.ops.permuteOutputVar(x_model_backbone__76, [3 4 2 1], 4);

            x_model_backbone__76 = dlarray(single(x_model_backbone__76), 'SSCB');
        end

        function [x_model_backbone__76, x_model_backbone__76NumDims1017] = ReduceMeanGraph1015(this, x_model_backbone__70, x_model_backbone__70NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1016, x_model_backbone__70NumDims);
            xMean = mean(x_model_backbone__70, dims);
            x_model_backbone__76 = xMean;
            x_model_backbone__76NumDims = x_model_backbone__70NumDims;

            % Set graph output arguments
            x_model_backbone__76NumDims1017 = x_model_backbone__76NumDims;

        end

    end

end