classdef ReduceMeanLayer1030 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1030';
        end
    end


    methods
        function this = ReduceMeanLayer1030(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone_446'};
        end

        function [x_model_backbone_446] = predict(this, x_model_backbone_440)
            if isdlarray(x_model_backbone_440)
                x_model_backbone_440 = stripdims(x_model_backbone_440);
            end
            x_model_backbone_440NumDims = 4;
            x_model_backbone_440 = netrai_fold0.ops.permuteInputVar(x_model_backbone_440, [4 3 1 2], 4);

            [x_model_backbone_446, x_model_backbone_446NumDims] = ReduceMeanGraph1090(this, x_model_backbone_440, x_model_backbone_440NumDims, false);
            x_model_backbone_446 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_446, [3 4 2 1], 4);

            x_model_backbone_446 = dlarray(single(x_model_backbone_446), 'SSCB');
        end

        function [x_model_backbone_446] = forward(this, x_model_backbone_440)
            if isdlarray(x_model_backbone_440)
                x_model_backbone_440 = stripdims(x_model_backbone_440);
            end
            x_model_backbone_440NumDims = 4;
            x_model_backbone_440 = netrai_fold0.ops.permuteInputVar(x_model_backbone_440, [4 3 1 2], 4);

            [x_model_backbone_446, x_model_backbone_446NumDims] = ReduceMeanGraph1090(this, x_model_backbone_440, x_model_backbone_440NumDims, true);
            x_model_backbone_446 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_446, [3 4 2 1], 4);

            x_model_backbone_446 = dlarray(single(x_model_backbone_446), 'SSCB');
        end

        function [x_model_backbone_446, x_model_backbone_446NumDims1092] = ReduceMeanGraph1090(this, x_model_backbone_440, x_model_backbone_440NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1091, x_model_backbone_440NumDims);
            xMean = mean(x_model_backbone_440, dims);
            x_model_backbone_446 = xMean;
            x_model_backbone_446NumDims = x_model_backbone_440NumDims;

            % Set graph output arguments
            x_model_backbone_446NumDims1092 = x_model_backbone_446NumDims;

        end

    end

end