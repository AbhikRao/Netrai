classdef ReduceMeanLayer1009 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1009';
        end
    end


    methods
        function this = ReduceMeanLayer1009(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone_135'};
        end

        function [x_model_backbone_135] = predict(this, x_model_backbone_129)
            if isdlarray(x_model_backbone_129)
                x_model_backbone_129 = stripdims(x_model_backbone_129);
            end
            x_model_backbone_129NumDims = 4;
            x_model_backbone_129 = netrai_fold0.ops.permuteInputVar(x_model_backbone_129, [4 3 1 2], 4);

            [x_model_backbone_135, x_model_backbone_135NumDims] = ReduceMeanGraph1027(this, x_model_backbone_129, x_model_backbone_129NumDims, false);
            x_model_backbone_135 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_135, [3 4 2 1], 4);

            x_model_backbone_135 = dlarray(single(x_model_backbone_135), 'SSCB');
        end

        function [x_model_backbone_135] = forward(this, x_model_backbone_129)
            if isdlarray(x_model_backbone_129)
                x_model_backbone_129 = stripdims(x_model_backbone_129);
            end
            x_model_backbone_129NumDims = 4;
            x_model_backbone_129 = netrai_fold0.ops.permuteInputVar(x_model_backbone_129, [4 3 1 2], 4);

            [x_model_backbone_135, x_model_backbone_135NumDims] = ReduceMeanGraph1027(this, x_model_backbone_129, x_model_backbone_129NumDims, true);
            x_model_backbone_135 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_135, [3 4 2 1], 4);

            x_model_backbone_135 = dlarray(single(x_model_backbone_135), 'SSCB');
        end

        function [x_model_backbone_135, x_model_backbone_135NumDims1029] = ReduceMeanGraph1027(this, x_model_backbone_129, x_model_backbone_129NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1028, x_model_backbone_129NumDims);
            xMean = mean(x_model_backbone_129, dims);
            x_model_backbone_135 = xMean;
            x_model_backbone_135NumDims = x_model_backbone_129NumDims;

            % Set graph output arguments
            x_model_backbone_135NumDims1029 = x_model_backbone_135NumDims;

        end

    end

end