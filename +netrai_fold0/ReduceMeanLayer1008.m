classdef ReduceMeanLayer1008 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1008';
        end
    end


    methods
        function this = ReduceMeanLayer1008(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone_120'};
        end

        function [x_model_backbone_120] = predict(this, x_model_backbone_114)
            if isdlarray(x_model_backbone_114)
                x_model_backbone_114 = stripdims(x_model_backbone_114);
            end
            x_model_backbone_114NumDims = 4;
            x_model_backbone_114 = netrai_fold0.ops.permuteInputVar(x_model_backbone_114, [4 3 1 2], 4);

            [x_model_backbone_120, x_model_backbone_120NumDims] = ReduceMeanGraph1024(this, x_model_backbone_114, x_model_backbone_114NumDims, false);
            x_model_backbone_120 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_120, [3 4 2 1], 4);

            x_model_backbone_120 = dlarray(single(x_model_backbone_120), 'SSCB');
        end

        function [x_model_backbone_120] = forward(this, x_model_backbone_114)
            if isdlarray(x_model_backbone_114)
                x_model_backbone_114 = stripdims(x_model_backbone_114);
            end
            x_model_backbone_114NumDims = 4;
            x_model_backbone_114 = netrai_fold0.ops.permuteInputVar(x_model_backbone_114, [4 3 1 2], 4);

            [x_model_backbone_120, x_model_backbone_120NumDims] = ReduceMeanGraph1024(this, x_model_backbone_114, x_model_backbone_114NumDims, true);
            x_model_backbone_120 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_120, [3 4 2 1], 4);

            x_model_backbone_120 = dlarray(single(x_model_backbone_120), 'SSCB');
        end

        function [x_model_backbone_120, x_model_backbone_120NumDims1026] = ReduceMeanGraph1024(this, x_model_backbone_114, x_model_backbone_114NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1025, x_model_backbone_114NumDims);
            xMean = mean(x_model_backbone_114, dims);
            x_model_backbone_120 = xMean;
            x_model_backbone_120NumDims = x_model_backbone_114NumDims;

            % Set graph output arguments
            x_model_backbone_120NumDims1026 = x_model_backbone_120NumDims;

        end

    end

end