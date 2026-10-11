classdef ReduceMeanLayer1002 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1002';
        end
    end


    methods
        function this = ReduceMeanLayer1002(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone__31'};
        end

        function [x_model_backbone__31] = predict(this, x_model_backbone__25)
            if isdlarray(x_model_backbone__25)
                x_model_backbone__25 = stripdims(x_model_backbone__25);
            end
            x_model_backbone__25NumDims = 4;
            x_model_backbone__25 = netrai_fold0.ops.permuteInputVar(x_model_backbone__25, [4 3 1 2], 4);

            [x_model_backbone__31, x_model_backbone__31NumDims] = ReduceMeanGraph1006(this, x_model_backbone__25, x_model_backbone__25NumDims, false);
            x_model_backbone__31 = netrai_fold0.ops.permuteOutputVar(x_model_backbone__31, [3 4 2 1], 4);

            x_model_backbone__31 = dlarray(single(x_model_backbone__31), 'SSCB');
        end

        function [x_model_backbone__31] = forward(this, x_model_backbone__25)
            if isdlarray(x_model_backbone__25)
                x_model_backbone__25 = stripdims(x_model_backbone__25);
            end
            x_model_backbone__25NumDims = 4;
            x_model_backbone__25 = netrai_fold0.ops.permuteInputVar(x_model_backbone__25, [4 3 1 2], 4);

            [x_model_backbone__31, x_model_backbone__31NumDims] = ReduceMeanGraph1006(this, x_model_backbone__25, x_model_backbone__25NumDims, true);
            x_model_backbone__31 = netrai_fold0.ops.permuteOutputVar(x_model_backbone__31, [3 4 2 1], 4);

            x_model_backbone__31 = dlarray(single(x_model_backbone__31), 'SSCB');
        end

        function [x_model_backbone__31, x_model_backbone__31NumDims1008] = ReduceMeanGraph1006(this, x_model_backbone__25, x_model_backbone__25NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1007, x_model_backbone__25NumDims);
            xMean = mean(x_model_backbone__25, dims);
            x_model_backbone__31 = xMean;
            x_model_backbone__31NumDims = x_model_backbone__25NumDims;

            % Set graph output arguments
            x_model_backbone__31NumDims1008 = x_model_backbone__31NumDims;

        end

    end

end