classdef ReduceMeanLayer1021 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'netrai_fold0.coder.ReduceMeanLayer1021';
        end
    end


    methods
        function this = ReduceMeanLayer1021(name)
            this.Name = name;
            this.OutputNames = {'x_model_backbone_313'};
        end

        function [x_model_backbone_313] = predict(this, x_model_backbone_307)
            if isdlarray(x_model_backbone_307)
                x_model_backbone_307 = stripdims(x_model_backbone_307);
            end
            x_model_backbone_307NumDims = 4;
            x_model_backbone_307 = netrai_fold0.ops.permuteInputVar(x_model_backbone_307, [4 3 1 2], 4);

            [x_model_backbone_313, x_model_backbone_313NumDims] = ReduceMeanGraph1063(this, x_model_backbone_307, x_model_backbone_307NumDims, false);
            x_model_backbone_313 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_313, [3 4 2 1], 4);

            x_model_backbone_313 = dlarray(single(x_model_backbone_313), 'SSCB');
        end

        function [x_model_backbone_313] = forward(this, x_model_backbone_307)
            if isdlarray(x_model_backbone_307)
                x_model_backbone_307 = stripdims(x_model_backbone_307);
            end
            x_model_backbone_307NumDims = 4;
            x_model_backbone_307 = netrai_fold0.ops.permuteInputVar(x_model_backbone_307, [4 3 1 2], 4);

            [x_model_backbone_313, x_model_backbone_313NumDims] = ReduceMeanGraph1063(this, x_model_backbone_307, x_model_backbone_307NumDims, true);
            x_model_backbone_313 = netrai_fold0.ops.permuteOutputVar(x_model_backbone_313, [3 4 2 1], 4);

            x_model_backbone_313 = dlarray(single(x_model_backbone_313), 'SSCB');
        end

        function [x_model_backbone_313, x_model_backbone_313NumDims1065] = ReduceMeanGraph1063(this, x_model_backbone_307, x_model_backbone_307NumDims, Training)

            % Execute the operators:
            % ReduceMean:
            dims = netrai_fold0.ops.prepareReduceArgs(this.Vars.ReduceMeanAxes1064, x_model_backbone_307NumDims);
            xMean = mean(x_model_backbone_307, dims);
            x_model_backbone_313 = xMean;
            x_model_backbone_313NumDims = x_model_backbone_307NumDims;

            % Set graph output arguments
            x_model_backbone_313NumDims1065 = x_model_backbone_313NumDims;

        end

    end

end