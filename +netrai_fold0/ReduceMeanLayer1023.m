classdef ReduceMeanLayer1023 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1023';
        end
    end


    methods
        function this = ReduceMeanLayer1023(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone_342'};
        end

        function [x_model_backbone_342] = predict(this, x_model_backbone_336)
            if isdlarray(x_model_backbone_336)
                x_model_backbone_336 = stripdims(x_model_backbone_336);
            end
            x_model_backbone_336NumDims = 4;
            x_model_backbone_336 = netrai_fold0.ops.permuteInputVar(x_model_backbone_336, [4 3 1 2], 4);

            [x_model_backbone_342, x_model_backbone_342NumDims] = ReduceMeanGraph1069(this, x_model_backbone_336, x_model_backbone_336NumDims, false);
            x_model_backbone_342 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_342, [3 4 2 1], 4);

            x_model_backbone_342 = dlarray(single(x_model_backbone_342), 'SSCB');
        end

        function [x_model_backbone_342] = forward(this, x_model_backbone_336)
            if isdlarray(x_model_backbone_336)
                x_model_backbone_336 = stripdims(x_model_backbone_336);
            end
            x_model_backbone_336NumDims = 4;
            x_model_backbone_336 = netrai_fold0.ops.permuteInputVar(x_model_backbone_336, [4 3 1 2], 4);

            [x_model_backbone_342, x_model_backbone_342NumDims] = ReduceMeanGraph1069(this, x_model_backbone_336, x_model_backbone_336NumDims, true);
            x_model_backbone_342 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_342, [3 4 2 1], 4);

            x_model_backbone_342 = dlarray(single(x_model_backbone_342), 'SSCB');
        end

        function [x_model_backbone_342, x_model_backbone_342NumDims1071] = ReduceMeanGraph1069(this, x_model_backbone_336, x_model_backbone_336NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1070, x_model_backbone_336NumDims);
            xMean = mean(x_model_backbone_336, dims);
            x_model_backbone_342 = xMean;
            x_model_backbone_342NumDims = x_model_backbone_336NumDims;

            % Set graph output arguments
            x_model_backbone_342NumDims1071 = x_model_backbone_342NumDims;

        end

    end

end