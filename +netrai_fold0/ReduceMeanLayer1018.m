classdef ReduceMeanLayer1018 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1018';
        end
    end


    methods
        function this = ReduceMeanLayer1018(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone_268'};
        end

        function [x_model_backbone_268] = predict(this, x_model_backbone_262)
            if isdlarray(x_model_backbone_262)
                x_model_backbone_262 = stripdims(x_model_backbone_262);
            end
            x_model_backbone_262NumDims = 4;
            x_model_backbone_262 = netrai_fold0.ops.permuteInputVar(x_model_backbone_262, [4 3 1 2], 4);

            [x_model_backbone_268, x_model_backbone_268NumDims] = ReduceMeanGraph1054(this, x_model_backbone_262, x_model_backbone_262NumDims, false);
            x_model_backbone_268 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_268, [3 4 2 1], 4);

            x_model_backbone_268 = dlarray(single(x_model_backbone_268), 'SSCB');
        end

        function [x_model_backbone_268] = forward(this, x_model_backbone_262)
            if isdlarray(x_model_backbone_262)
                x_model_backbone_262 = stripdims(x_model_backbone_262);
            end
            x_model_backbone_262NumDims = 4;
            x_model_backbone_262 = netrai_fold0.ops.permuteInputVar(x_model_backbone_262, [4 3 1 2], 4);

            [x_model_backbone_268, x_model_backbone_268NumDims] = ReduceMeanGraph1054(this, x_model_backbone_262, x_model_backbone_262NumDims, true);
            x_model_backbone_268 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_268, [3 4 2 1], 4);

            x_model_backbone_268 = dlarray(single(x_model_backbone_268), 'SSCB');
        end

        function [x_model_backbone_268, x_model_backbone_268NumDims1056] = ReduceMeanGraph1054(this, x_model_backbone_262, x_model_backbone_262NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1055, x_model_backbone_262NumDims);
            xMean = mean(x_model_backbone_262, dims);
            x_model_backbone_268 = xMean;
            x_model_backbone_268NumDims = x_model_backbone_262NumDims;

            % Set graph output arguments
            x_model_backbone_268NumDims1056 = x_model_backbone_268NumDims;

        end

    end

end