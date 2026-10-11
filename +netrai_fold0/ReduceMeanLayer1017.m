classdef ReduceMeanLayer1017 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1017';
        end
    end


    methods
        function this = ReduceMeanLayer1017(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone_253'};
        end

        function [x_model_backbone_253] = predict(this, x_model_backbone_247)
            if isdlarray(x_model_backbone_247)
                x_model_backbone_247 = stripdims(x_model_backbone_247);
            end
            x_model_backbone_247NumDims = 4;
            x_model_backbone_247 = netrai_fold0.ops.permuteInputVar(x_model_backbone_247, [4 3 1 2], 4);

            [x_model_backbone_253, x_model_backbone_253NumDims] = ReduceMeanGraph1051(this, x_model_backbone_247, x_model_backbone_247NumDims, false);
            x_model_backbone_253 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_253, [3 4 2 1], 4);

            x_model_backbone_253 = dlarray(single(x_model_backbone_253), 'SSCB');
        end

        function [x_model_backbone_253] = forward(this, x_model_backbone_247)
            if isdlarray(x_model_backbone_247)
                x_model_backbone_247 = stripdims(x_model_backbone_247);
            end
            x_model_backbone_247NumDims = 4;
            x_model_backbone_247 = netrai_fold0.ops.permuteInputVar(x_model_backbone_247, [4 3 1 2], 4);

            [x_model_backbone_253, x_model_backbone_253NumDims] = ReduceMeanGraph1051(this, x_model_backbone_247, x_model_backbone_247NumDims, true);
            x_model_backbone_253 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_253, [3 4 2 1], 4);

            x_model_backbone_253 = dlarray(single(x_model_backbone_253), 'SSCB');
        end

        function [x_model_backbone_253, x_model_backbone_253NumDims1053] = ReduceMeanGraph1051(this, x_model_backbone_247, x_model_backbone_247NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1052, x_model_backbone_247NumDims);
            xMean = mean(x_model_backbone_247, dims);
            x_model_backbone_253 = xMean;
            x_model_backbone_253NumDims = x_model_backbone_247NumDims;

            % Set graph output arguments
            x_model_backbone_253NumDims1053 = x_model_backbone_253NumDims;

        end

    end

end