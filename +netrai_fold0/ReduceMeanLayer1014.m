classdef ReduceMeanLayer1014 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1014';
        end
    end


    methods
        function this = ReduceMeanLayer1014(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone_209'};
        end

        function [x_model_backbone_209] = predict(this, x_model_backbone_203)
            if isdlarray(x_model_backbone_203)
                x_model_backbone_203 = stripdims(x_model_backbone_203);
            end
            x_model_backbone_203NumDims = 4;
            x_model_backbone_203 = netrai_fold0.ops.permuteInputVar(x_model_backbone_203, [4 3 1 2], 4);

            [x_model_backbone_209, x_model_backbone_209NumDims] = ReduceMeanGraph1042(this, x_model_backbone_203, x_model_backbone_203NumDims, false);
            x_model_backbone_209 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_209, [3 4 2 1], 4);

            x_model_backbone_209 = dlarray(single(x_model_backbone_209), 'SSCB');
        end

        function [x_model_backbone_209] = forward(this, x_model_backbone_203)
            if isdlarray(x_model_backbone_203)
                x_model_backbone_203 = stripdims(x_model_backbone_203);
            end
            x_model_backbone_203NumDims = 4;
            x_model_backbone_203 = netrai_fold0.ops.permuteInputVar(x_model_backbone_203, [4 3 1 2], 4);

            [x_model_backbone_209, x_model_backbone_209NumDims] = ReduceMeanGraph1042(this, x_model_backbone_203, x_model_backbone_203NumDims, true);
            x_model_backbone_209 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_209, [3 4 2 1], 4);

            x_model_backbone_209 = dlarray(single(x_model_backbone_209), 'SSCB');
        end

        function [x_model_backbone_209, x_model_backbone_209NumDims1044] = ReduceMeanGraph1042(this, x_model_backbone_203, x_model_backbone_203NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1043, x_model_backbone_203NumDims);
            xMean = mean(x_model_backbone_203, dims);
            x_model_backbone_209 = xMean;
            x_model_backbone_209NumDims = x_model_backbone_203NumDims;

            % Set graph output arguments
            x_model_backbone_209NumDims1044 = x_model_backbone_209NumDims;

        end

    end

end