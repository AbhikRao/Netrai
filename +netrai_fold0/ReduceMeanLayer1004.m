classdef ReduceMeanLayer1004 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1004';
        end
    end


    methods
        function this = ReduceMeanLayer1004(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone__61'};
        end

        function [x_model_backbone__61] = predict(this, x_model_backbone__55)
            if isdlarray(x_model_backbone__55)
                x_model_backbone__55 = stripdims(x_model_backbone__55);
            end
            x_model_backbone__55NumDims = 4;
            x_model_backbone__55 = netrai_fold0.ops.permuteInputVar(x_model_backbone__55, [4 3 1 2], 4);

            [x_model_backbone__61, x_model_backbone__61NumDims] = ReduceMeanGraph1012(this, x_model_backbone__55, x_model_backbone__55NumDims, false);
            x_model_backbone__61 = netrai_fold0.ops.permuteOutputVar(x_model_backbone__61, [3 4 2 1], 4);

            x_model_backbone__61 = dlarray(single(x_model_backbone__61), 'SSCB');
        end

        function [x_model_backbone__61] = forward(this, x_model_backbone__55)
            if isdlarray(x_model_backbone__55)
                x_model_backbone__55 = stripdims(x_model_backbone__55);
            end
            x_model_backbone__55NumDims = 4;
            x_model_backbone__55 = netrai_fold0.ops.permuteInputVar(x_model_backbone__55, [4 3 1 2], 4);

            [x_model_backbone__61, x_model_backbone__61NumDims] = ReduceMeanGraph1012(this, x_model_backbone__55, x_model_backbone__55NumDims, true);
            x_model_backbone__61 = netrai_fold0.ops.permuteOutputVar(x_model_backbone__61, [3 4 2 1], 4);

            x_model_backbone__61 = dlarray(single(x_model_backbone__61), 'SSCB');
        end

        function [x_model_backbone__61, x_model_backbone__61NumDims1014] = ReduceMeanGraph1012(this, x_model_backbone__55, x_model_backbone__55NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1013, x_model_backbone__55NumDims);
            xMean = mean(x_model_backbone__55, dims);
            x_model_backbone__61 = xMean;
            x_model_backbone__61NumDims = x_model_backbone__55NumDims;

            % Set graph output arguments
            x_model_backbone__61NumDims1014 = x_model_backbone__61NumDims;

        end

    end

end