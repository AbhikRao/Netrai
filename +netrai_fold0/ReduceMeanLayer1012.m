classdef ReduceMeanLayer1012 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1012';
        end
    end


    methods
        function this = ReduceMeanLayer1012(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone_179'};
        end

        function [x_model_backbone_179] = predict(this, x_model_backbone_173)
            if isdlarray(x_model_backbone_173)
                x_model_backbone_173 = stripdims(x_model_backbone_173);
            end
            x_model_backbone_173NumDims = 4;
            x_model_backbone_173 = netrai_fold0.ops.permuteInputVar(x_model_backbone_173, [4 3 1 2], 4);

            [x_model_backbone_179, x_model_backbone_179NumDims] = ReduceMeanGraph1036(this, x_model_backbone_173, x_model_backbone_173NumDims, false);
            x_model_backbone_179 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_179, [3 4 2 1], 4);

            x_model_backbone_179 = dlarray(single(x_model_backbone_179), 'SSCB');
        end

        function [x_model_backbone_179] = forward(this, x_model_backbone_173)
            if isdlarray(x_model_backbone_173)
                x_model_backbone_173 = stripdims(x_model_backbone_173);
            end
            x_model_backbone_173NumDims = 4;
            x_model_backbone_173 = netrai_fold0.ops.permuteInputVar(x_model_backbone_173, [4 3 1 2], 4);

            [x_model_backbone_179, x_model_backbone_179NumDims] = ReduceMeanGraph1036(this, x_model_backbone_173, x_model_backbone_173NumDims, true);
            x_model_backbone_179 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_179, [3 4 2 1], 4);

            x_model_backbone_179 = dlarray(single(x_model_backbone_179), 'SSCB');
        end

        function [x_model_backbone_179, x_model_backbone_179NumDims1038] = ReduceMeanGraph1036(this, x_model_backbone_173, x_model_backbone_173NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1037, x_model_backbone_173NumDims);
            xMean = mean(x_model_backbone_173, dims);
            x_model_backbone_179 = xMean;
            x_model_backbone_179NumDims = x_model_backbone_173NumDims;

            % Set graph output arguments
            x_model_backbone_179NumDims1038 = x_model_backbone_179NumDims;

        end

    end

end