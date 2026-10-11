classdef ReduceMeanLayer1013 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1013';
        end
    end


    methods
        function this = ReduceMeanLayer1013(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone_194'};
        end

        function [x_model_backbone_194] = predict(this, x_model_backbone_188)
            if isdlarray(x_model_backbone_188)
                x_model_backbone_188 = stripdims(x_model_backbone_188);
            end
            x_model_backbone_188NumDims = 4;
            x_model_backbone_188 = netrai_fold0.ops.permuteInputVar(x_model_backbone_188, [4 3 1 2], 4);

            [x_model_backbone_194, x_model_backbone_194NumDims] = ReduceMeanGraph1039(this, x_model_backbone_188, x_model_backbone_188NumDims, false);
            x_model_backbone_194 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_194, [3 4 2 1], 4);

            x_model_backbone_194 = dlarray(single(x_model_backbone_194), 'SSCB');
        end

        function [x_model_backbone_194] = forward(this, x_model_backbone_188)
            if isdlarray(x_model_backbone_188)
                x_model_backbone_188 = stripdims(x_model_backbone_188);
            end
            x_model_backbone_188NumDims = 4;
            x_model_backbone_188 = netrai_fold0.ops.permuteInputVar(x_model_backbone_188, [4 3 1 2], 4);

            [x_model_backbone_194, x_model_backbone_194NumDims] = ReduceMeanGraph1039(this, x_model_backbone_188, x_model_backbone_188NumDims, true);
            x_model_backbone_194 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_194, [3 4 2 1], 4);

            x_model_backbone_194 = dlarray(single(x_model_backbone_194), 'SSCB');
        end

        function [x_model_backbone_194, x_model_backbone_194NumDims1041] = ReduceMeanGraph1039(this, x_model_backbone_188, x_model_backbone_188NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1040, x_model_backbone_188NumDims);
            xMean = mean(x_model_backbone_188, dims);
            x_model_backbone_194 = xMean;
            x_model_backbone_194NumDims = x_model_backbone_188NumDims;

            % Set graph output arguments
            x_model_backbone_194NumDims1041 = x_model_backbone_194NumDims;

        end

    end

end