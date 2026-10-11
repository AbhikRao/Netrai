classdef ReduceMeanLayer1011 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1011';
        end
    end


    methods
        function this = ReduceMeanLayer1011(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone_164'};
        end

        function [x_model_backbone_164] = predict(this, x_model_backbone_158)
            if isdlarray(x_model_backbone_158)
                x_model_backbone_158 = stripdims(x_model_backbone_158);
            end
            x_model_backbone_158NumDims = 4;
            x_model_backbone_158 = netrai_fold0.ops.permuteInputVar(x_model_backbone_158, [4 3 1 2], 4);

            [x_model_backbone_164, x_model_backbone_164NumDims] = ReduceMeanGraph1033(this, x_model_backbone_158, x_model_backbone_158NumDims, false);
            x_model_backbone_164 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_164, [3 4 2 1], 4);

            x_model_backbone_164 = dlarray(single(x_model_backbone_164), 'SSCB');
        end

        function [x_model_backbone_164] = forward(this, x_model_backbone_158)
            if isdlarray(x_model_backbone_158)
                x_model_backbone_158 = stripdims(x_model_backbone_158);
            end
            x_model_backbone_158NumDims = 4;
            x_model_backbone_158 = netrai_fold0.ops.permuteInputVar(x_model_backbone_158, [4 3 1 2], 4);

            [x_model_backbone_164, x_model_backbone_164NumDims] = ReduceMeanGraph1033(this, x_model_backbone_158, x_model_backbone_158NumDims, true);
            x_model_backbone_164 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_164, [3 4 2 1], 4);

            x_model_backbone_164 = dlarray(single(x_model_backbone_164), 'SSCB');
        end

        function [x_model_backbone_164, x_model_backbone_164NumDims1035] = ReduceMeanGraph1033(this, x_model_backbone_158, x_model_backbone_158NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1034, x_model_backbone_158NumDims);
            xMean = mean(x_model_backbone_158, dims);
            x_model_backbone_164 = xMean;
            x_model_backbone_164NumDims = x_model_backbone_158NumDims;

            % Set graph output arguments
            x_model_backbone_164NumDims1035 = x_model_backbone_164NumDims;

        end

    end

end