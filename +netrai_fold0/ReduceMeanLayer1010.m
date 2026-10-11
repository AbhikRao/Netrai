classdef ReduceMeanLayer1010 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1010';
        end
    end


    methods
        function this = ReduceMeanLayer1010(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone_149'};
        end

        function [x_model_backbone_149] = predict(this, x_model_backbone_143)
            if isdlarray(x_model_backbone_143)
                x_model_backbone_143 = stripdims(x_model_backbone_143);
            end
            x_model_backbone_143NumDims = 4;
            x_model_backbone_143 = netrai_fold0.ops.permuteInputVar(x_model_backbone_143, [4 3 1 2], 4);

            [x_model_backbone_149, x_model_backbone_149NumDims] = ReduceMeanGraph1030(this, x_model_backbone_143, x_model_backbone_143NumDims, false);
            x_model_backbone_149 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_149, [3 4 2 1], 4);

            x_model_backbone_149 = dlarray(single(x_model_backbone_149), 'SSCB');
        end

        function [x_model_backbone_149] = forward(this, x_model_backbone_143)
            if isdlarray(x_model_backbone_143)
                x_model_backbone_143 = stripdims(x_model_backbone_143);
            end
            x_model_backbone_143NumDims = 4;
            x_model_backbone_143 = netrai_fold0.ops.permuteInputVar(x_model_backbone_143, [4 3 1 2], 4);

            [x_model_backbone_149, x_model_backbone_149NumDims] = ReduceMeanGraph1030(this, x_model_backbone_143, x_model_backbone_143NumDims, true);
            x_model_backbone_149 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_149, [3 4 2 1], 4);

            x_model_backbone_149 = dlarray(single(x_model_backbone_149), 'SSCB');
        end

        function [x_model_backbone_149, x_model_backbone_149NumDims1032] = ReduceMeanGraph1030(this, x_model_backbone_143, x_model_backbone_143NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1031, x_model_backbone_143NumDims);
            xMean = mean(x_model_backbone_143, dims);
            x_model_backbone_149 = xMean;
            x_model_backbone_149NumDims = x_model_backbone_143NumDims;

            % Set graph output arguments
            x_model_backbone_149NumDims1032 = x_model_backbone_149NumDims;

        end

    end

end