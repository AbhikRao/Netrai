classdef ReduceMeanLayer1006 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1006';
        end
    end


    methods
        function this = ReduceMeanLayer1006(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone__90'};
        end

        function [x_model_backbone__90] = predict(this, x_model_backbone__84)
            if isdlarray(x_model_backbone__84)
                x_model_backbone__84 = stripdims(x_model_backbone__84);
            end
            x_model_backbone__84NumDims = 4;
            x_model_backbone__84 = netrai_fold0.ops.permuteInputVar(x_model_backbone__84, [4 3 1 2], 4);

            [x_model_backbone__90, x_model_backbone__90NumDims] = ReduceMeanGraph1018(this, x_model_backbone__84, x_model_backbone__84NumDims, false);
            x_model_backbone__90 = netrai_fold0.ops.permuteOutputVar(x_model_backbone__90, [3 4 2 1], 4);

            x_model_backbone__90 = dlarray(single(x_model_backbone__90), 'SSCB');
        end

        function [x_model_backbone__90] = forward(this, x_model_backbone__84)
            if isdlarray(x_model_backbone__84)
                x_model_backbone__84 = stripdims(x_model_backbone__84);
            end
            x_model_backbone__84NumDims = 4;
            x_model_backbone__84 = netrai_fold0.ops.permuteInputVar(x_model_backbone__84, [4 3 1 2], 4);

            [x_model_backbone__90, x_model_backbone__90NumDims] = ReduceMeanGraph1018(this, x_model_backbone__84, x_model_backbone__84NumDims, true);
            x_model_backbone__90 = netrai_fold0.ops.permuteOutputVar(x_model_backbone__90, [3 4 2 1], 4);

            x_model_backbone__90 = dlarray(single(x_model_backbone__90), 'SSCB');
        end

        function [x_model_backbone__90, x_model_backbone__90NumDims1020] = ReduceMeanGraph1018(this, x_model_backbone__84, x_model_backbone__84NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1019, x_model_backbone__84NumDims);
            xMean = mean(x_model_backbone__84, dims);
            x_model_backbone__90 = xMean;
            x_model_backbone__90NumDims = x_model_backbone__84NumDims;

            % Set graph output arguments
            x_model_backbone__90NumDims1020 = x_model_backbone__90NumDims;

        end

    end

end