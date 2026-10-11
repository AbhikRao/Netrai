classdef ReduceMeanLayer1001 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1001';
        end
    end


    methods
        function this = ReduceMeanLayer1001(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone__17'};
        end

        function [x_model_backbone__17] = predict(this, x_model_backbone__12)
            if isdlarray(x_model_backbone__12)
                x_model_backbone__12 = stripdims(x_model_backbone__12);
            end
            x_model_backbone__12NumDims = 4;
            x_model_backbone__12 = netrai_fold0.ops.permuteInputVar(x_model_backbone__12, [4 3 1 2], 4);

            [x_model_backbone__17, x_model_backbone__17NumDims] = ReduceMeanGraph1003(this, x_model_backbone__12, x_model_backbone__12NumDims, false);
            x_model_backbone__17 = netrai_fold0.ops.permuteOutputVar(x_model_backbone__17, [3 4 2 1], 4);

            x_model_backbone__17 = dlarray(single(x_model_backbone__17), 'SSCB');
        end

        function [x_model_backbone__17] = forward(this, x_model_backbone__12)
            if isdlarray(x_model_backbone__12)
                x_model_backbone__12 = stripdims(x_model_backbone__12);
            end
            x_model_backbone__12NumDims = 4;
            x_model_backbone__12 = netrai_fold0.ops.permuteInputVar(x_model_backbone__12, [4 3 1 2], 4);

            [x_model_backbone__17, x_model_backbone__17NumDims] = ReduceMeanGraph1003(this, x_model_backbone__12, x_model_backbone__12NumDims, true);
            x_model_backbone__17 = netrai_fold0.ops.permuteOutputVar(x_model_backbone__17, [3 4 2 1], 4);

            x_model_backbone__17 = dlarray(single(x_model_backbone__17), 'SSCB');
        end

        function [x_model_backbone__17, x_model_backbone__17NumDims1005] = ReduceMeanGraph1003(this, x_model_backbone__12, x_model_backbone__12NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1004, x_model_backbone__12NumDims);
            xMean = mean(x_model_backbone__12, dims);
            x_model_backbone__17 = xMean;
            x_model_backbone__17NumDims = x_model_backbone__12NumDims;

            % Set graph output arguments
            x_model_backbone__17NumDims1005 = x_model_backbone__17NumDims;

        end

    end

end